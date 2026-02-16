package pdc;

import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.Socket;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * A Worker is a node in the cluster capable of high-concurrency computation.
 * 
 * RPC Worker Architecture:
 * - Connects to Master via TCP socket using custom RPC protocol
 * - Sends periodic heartbeats to prove liveness
 * - Receives RPC task assignment requests and processes them concurrently
 * - Returns RPC responses with results to Master asynchronously
 * - Uses internal thread pool for parallel task execution
 * - Handles multiple RPC requests concurrently for maximum throughput
 */
public class Worker {
    
    private final String workerId;
    private final String masterHost;
    private final int masterPort;
    
    private Socket socket;
    private InputStream inputStream;
    private OutputStream outputStream;
    
    private final ExecutorService taskExecutor;
    private final ExecutorService heartbeatExecutor;
    private final AtomicBoolean running;
    
    private static final int HEARTBEAT_INTERVAL_MS = 2000; // 2 seconds
    private static final int TASK_THREAD_POOL_SIZE = 4;
    
    /**
     * Creates a worker with specified ID.
     */
    public Worker() {
        // Read configuration from environment or use defaults
        this.workerId = System.getenv("WORKER_ID") != null ? 
            System.getenv("WORKER_ID") : "worker-" + System.currentTimeMillis();
        this.masterHost = System.getenv("MASTER_HOST") != null ? 
            System.getenv("MASTER_HOST") : "localhost";
        this.masterPort = System.getenv("MASTER_PORT") != null ? 
            Integer.parseInt(System.getenv("MASTER_PORT")) : 9999;
        
        this.taskExecutor = Executors.newFixedThreadPool(TASK_THREAD_POOL_SIZE);
        this.heartbeatExecutor = Executors.newSingleThreadExecutor();
        this.running = new AtomicBoolean(false);
    }
    
    /**
     * Constructor for custom configuration.
     */
    public Worker(String workerId, String masterHost, int masterPort) {
        this.workerId = workerId;
        this.masterHost = masterHost;
        this.masterPort = masterPort;
        
        this.taskExecutor = Executors.newFixedThreadPool(TASK_THREAD_POOL_SIZE);
        this.heartbeatExecutor = Executors.newSingleThreadExecutor();
        this.running = new AtomicBoolean(false);
    }

    /**
     * Connects to the Master and initiates the registration handshake.
     */
    public void joinCluster(String masterHost, int port) {
        try {
            System.err.println("[Worker " + workerId + "] Connecting to master at " + 
                masterHost + ":" + port);
            
            socket = new Socket(masterHost, port);
            inputStream = socket.getInputStream();
            outputStream = socket.getOutputStream();
            
            // Send registration message
            Message registerMsg = new Message(Message.MessageType.REGISTER, workerId);
            sendMessage(registerMsg);
            
            System.err.println("[Worker " + workerId + "] Registration sent");
            
        } catch (IOException e) {
            System.err.println("[Worker " + workerId + "] Failed to connect: " + e.getMessage());
            // Don't throw - handle gracefully for testing
        }
    }

    /**
     * Main execution loop - RPC request handler.
     * Starts heartbeat thread and listens for RPC task assignment requests.
     */
    public void execute() {
        // If not already connected, try to connect
        if (socket == null || !socket.isConnected()) {
            joinCluster(masterHost, masterPort);
        }
        
        if (socket == null || !socket.isConnected()) {
            System.err.println("[Worker " + workerId + "] Cannot execute - not connected");
            return;
        }
        
        running.set(true);
        
        // Start heartbeat thread
        startHeartbeat();
        
        // Main message processing loop
        System.err.println("[Worker " + workerId + "] Starting execution loop");
        
        try {
            while (running.get()) {
                // Read message from master
                byte[] messageBytes = Message.readMessageFromStream(inputStream);
                Message message = Message.unpack(messageBytes);
                
                // Handle message based on type
                handleMessage(message);
            }
        } catch (EOFException e) {
            System.err.println("[Worker " + workerId + "] Master disconnected");
        } catch (IOException e) {
            System.err.println("[Worker " + workerId + "] Communication error: " + e.getMessage());
        } finally {
            shutdown();
        }
    }
    
    /**
     * Handles incoming RPC messages from master.
     * Processes RPC requests and dispatches to appropriate handlers.
     */
    private void handleMessage(Message message) {
        switch (message.getMessageType()) {
            case TASK_ASSIGNMENT:
                handleTaskAssignment(message);
                break;
                
            case SHUTDOWN:
                System.err.println("[Worker " + workerId + "] Received shutdown command");
                running.set(false);
                break;
                
            case ACK:
                // Registration acknowledged
                System.err.println("[Worker " + workerId + "] Registration acknowledged");
                break;
                
            default:
                System.err.println("[Worker " + workerId + "] Unknown message type: " + 
                    message.getMessageType());
        }
    }
    
    /**
     * Handles task assignment by submitting to thread pool.
     */
    private void handleTaskAssignment(Message message) {
        long taskId = message.getTaskId();
        System.err.println("[Worker " + workerId + "] Received task " + taskId + 
            " [rows " + message.getStartRow() + "-" + message.getEndRow() + "]");
        
        // Submit task to thread pool for parallel execution
        taskExecutor.submit(() -> {
            try {
                executeTask(message);
            } catch (Exception e) {
                System.err.println("[Worker " + workerId + "] Task " + taskId + 
                    " failed: " + e.getMessage());
                e.printStackTrace();
            }
        });
    }
    
    /**
     * Executes a matrix multiplication task.
     * Computes partial result for assigned rows.
     */
    private void executeTask(Message taskMessage) throws IOException {
        long taskId = taskMessage.getTaskId();
        int startRow = taskMessage.getStartRow();
        int endRow = taskMessage.getEndRow();
        int[][] matrixA = taskMessage.getMatrixA();
        int[][] matrixB = taskMessage.getMatrixB();
        
        long startTime = System.currentTimeMillis();
        
        // Validate input
        if (matrixA == null || matrixB == null) {
            throw new IllegalArgumentException("Null matrices in task " + taskId);
        }
        
        int rowsA = matrixA.length;
        int colsA = matrixA[0].length;
        int colsB = matrixB[0].length;
        
        // Compute partial result (only assigned rows)
        int resultRows = endRow - startRow;
        int[][] result = new int[resultRows][colsB];
        
        for (int i = 0; i < resultRows; i++) {
            int actualRow = startRow + i;
            if (actualRow >= rowsA) break; // Safety check
            
            for (int j = 0; j < colsB; j++) {
                int sum = 0;
                for (int k = 0; k < colsA; k++) {
                    sum += matrixA[actualRow][k] * matrixB[k][j];
                }
                result[i][j] = sum;
            }
        }
        
        long duration = System.currentTimeMillis() - startTime;
        System.err.println("[Worker " + workerId + "] Task " + taskId + 
            " completed in " + duration + "ms");
        
        // Send result back to master
        Message resultMsg = new Message(Message.MessageType.TASK_RESULT, workerId);
        resultMsg.setTaskId(taskId);
        resultMsg.setResultMatrix(result);
        
        synchronized (outputStream) {
            sendMessage(resultMsg);
        }
    }
    
    /**
     * Starts the heartbeat thread.
     */
    private void startHeartbeat() {
        heartbeatExecutor.submit(() -> {
            while (running.get()) {
                try {
                    Thread.sleep(HEARTBEAT_INTERVAL_MS);
                    
                    Message heartbeat = new Message(Message.MessageType.HEARTBEAT, workerId);
                    
                    synchronized (outputStream) {
                        sendMessage(heartbeat);
                    }
                    
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                } catch (IOException e) {
                    System.err.println("[Worker " + workerId + "] Heartbeat failed: " + 
                        e.getMessage());
                    running.set(false);
                    break;
                }
            }
        });
    }
    
    /**
     * Sends a message to the master.
     */
    private void sendMessage(Message message) throws IOException {
        byte[] messageBytes = message.pack();
        outputStream.write(messageBytes);
        outputStream.flush();
    }
    
    /**
     * Graceful shutdown.
     */
    private void shutdown() {
        running.set(false);
        
        System.err.println("[Worker " + workerId + "] Shutting down...");
        
        // Shutdown executors
        heartbeatExecutor.shutdownNow();
        taskExecutor.shutdown();
        try {
            // Wait for shutdown with timeout
            long deadline = System.currentTimeMillis() + 5000;
            while (!taskExecutor.isShutdown() && System.currentTimeMillis() < deadline) {
                Thread.sleep(100);
            }
            taskExecutor.shutdownNow();
        } catch (InterruptedException e) {
            taskExecutor.shutdownNow();
        }
        
        // Close socket
        try {
            if (socket != null && !socket.isClosed()) {
                socket.close();
            }
        } catch (IOException e) {
            System.err.println("[Worker " + workerId + "] Error closing socket: " + 
                e.getMessage());
        }
        
        System.err.println("[Worker " + workerId + "] Shutdown complete");
    }
    
    /**
     * Main entry point for running worker as standalone process.
     */
    public static void main(String[] args) {
        Worker worker = new Worker();
        
        // Add shutdown hook
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.err.println("[Worker] Shutdown hook triggered");
            worker.shutdown();
        }));
        
        // Execute worker
        worker.execute();
    }
}
