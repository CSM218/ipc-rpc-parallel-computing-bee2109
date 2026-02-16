package pdc;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;

/**
 * Message represents the communication unit in the CSM218 protocol.
 * 
 * This class provides RPC (Remote Procedure Call) abstraction for distributed
 * communication between Master and Worker nodes.
 * 
 * Wire Format (Length-Prefixed Framing):
 * +----------+----------+----------+----------+-----------------+
 * |  MAGIC   |  LENGTH  |   TYPE   | RESERVED |    PAYLOAD      |
 * | 4 bytes  | 4 bytes  |  1 byte  |  3 bytes |  Variable       |
 * +----------+----------+----------+----------+-----------------+
 * 
 * MAGIC: 0x43534D32 ("CSM2" in ASCII) - Protocol validation
 * LENGTH: Payload size in bytes (excludes header)
 * TYPE: Message type enum (see MessageType)
 * RESERVED: 3 bytes for future protocol extensions
 * PAYLOAD: Message-specific data
 */
public class Message {
    // Protocol Constants
    private static final int MAGIC_NUMBER = 0x43534D32; // "CSM2"
    private static final int HEADER_SIZE = 12; // 4 + 4 + 1 + 3
    private static final int MAX_PAYLOAD_SIZE = 50 * 1024 * 1024; // 50MB safety limit
    
    // Message Types
    public static enum MessageType {
        REGISTER(0x01),
        TASK_ASSIGNMENT(0x02),
        TASK_RESULT(0x03),
        HEARTBEAT(0x04),
        SHUTDOWN(0x05),
        ACK(0x06),
        ERROR(0x07);
        
        private final byte code;
        MessageType(int code) { this.code = (byte) code; }
        public byte getCode() { return code; }
        
        public static MessageType fromCode(byte code) {
            for (MessageType type : values()) {
                if (type.code == code) return type;
            }
            throw new IllegalArgumentException("Unknown message type: " + code);
        }
    }
    
    // Message Fields
    public String magic;
    public int version;
    public String type;
    public String messageType;
    public String studentId;
    public String sender;
    public long timestamp;
    public byte[] payload;
    
    // Structured fields for typed access
    private MessageType messageTypeEnum;
    private long taskId;
    private int startRow;
    private int endRow;
    private int[][] matrixA;
    private int[][] matrixB;
    private int[][] resultMatrix;
    private String workerId;

    public Message() {
        this.timestamp = System.currentTimeMillis();
        this.magic = "CSM218";
        this.version = 1;
        this.studentId = System.getenv("STUDENT_ID") != null ? 
            System.getenv("STUDENT_ID") : "N02211720B";
    }
    
    public Message(MessageType type, String workerId) {
        this();
        this.messageTypeEnum = type;
        this.workerId = workerId;
        this.type = type.name();
        this.messageType = type.name();
    }

    // Getters and Setters
    public MessageType getMessageType() { return messageTypeEnum; }
    public void setMessageType(MessageType type) { 
        this.messageTypeEnum = type; 
        this.type = type.name();
        this.messageType = type.name();
    }
    
    public long getTaskId() { return taskId; }
    public void setTaskId(long taskId) { this.taskId = taskId; }
    
    public int getStartRow() { return startRow; }
    public void setStartRow(int row) { this.startRow = row; }
    
    public int getEndRow() { return endRow; }
    public void setEndRow(int row) { this.endRow = row; }
    
    public int[][] getMatrixA() { return matrixA; }
    public void setMatrixA(int[][] matrix) { this.matrixA = matrix; }
    
    public int[][] getMatrixB() { return matrixB; }
    public void setMatrixB(int[][] matrix) { this.matrixB = matrix; }
    
    public int[][] getResultMatrix() { return resultMatrix; }
    public void setResultMatrix(int[][] matrix) { this.resultMatrix = matrix; }
    
    public String getWorkerId() { return workerId; }
    public void setWorkerId(String id) { this.workerId = id; }

    /**
     * Converts the message to a byte stream for network transmission.
     * Implements length-prefixed framing to handle TCP stream boundaries.
     * 
     * This is the serialization method that packs the message into binary format.
     */
    public byte[] pack() throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(baos);
        
        // Build payload first to calculate length
        byte[] payloadBytes = buildPayload();
        
        // Write Header (12 bytes)
        dos.writeInt(MAGIC_NUMBER);              // 4 bytes: Magic
        dos.writeInt(payloadBytes.length);       // 4 bytes: Payload length
        dos.writeByte(messageTypeEnum.getCode());    // 1 byte: Message type
        dos.write(new byte[3]);                  // 3 bytes: Reserved (zeros)
        
        // Write Payload
        dos.write(payloadBytes);
        
        dos.flush();
        return baos.toByteArray();
    }

    /**
     * Serialize method (alias for pack).
     * Provides explicit serialization naming for protocol compliance.
     */
    public byte[] serialize() throws IOException {
        return pack();
    }

    /**
     * Reconstructs a Message from a byte stream.
     * Validates protocol magic and parses message-specific fields.
     * 
     * This is the deserialization method that unpacks binary data into a Message object.
     */
    public static Message unpack(byte[] data) throws IOException {
        if (data.length < HEADER_SIZE) {
            throw new ProtocolException("Message too short: " + data.length);
        }
        
        ByteBuffer buffer = ByteBuffer.wrap(data);
        
        // Parse Header
        int magic = buffer.getInt();
        if (magic != MAGIC_NUMBER) {
            throw new ProtocolException("Invalid magic: 0x" + Integer.toHexString(magic));
        }
        
        int payloadLength = buffer.getInt();
        if (payloadLength < 0 || payloadLength > MAX_PAYLOAD_SIZE) {
            throw new ProtocolException("Invalid payload length: " + payloadLength);
        }
        
        byte typeCode = buffer.get();
        MessageType type = MessageType.fromCode(typeCode);
        
        buffer.position(buffer.position() + 3); // Skip reserved bytes
        
        // Verify we have complete message
        if (data.length != HEADER_SIZE + payloadLength) {
            throw new ProtocolException("Incomplete message: expected " + 
                (HEADER_SIZE + payloadLength) + " bytes, got " + data.length);
        }
        
        // Extract payload
        byte[] payloadBytes = new byte[payloadLength];
        buffer.get(payloadBytes);
        
        // Parse payload based on type
        Message msg = new Message();
        msg.messageTypeEnum = type;
        msg.type = type.name();
        msg.messageType = type.name();
        msg.parsePayload(payloadBytes);
        
        return msg;
    }
    
    /**
     * Deserialize method (alias for unpack).
     * Provides explicit deserialization naming for protocol compliance.
     */
    public static Message deserialize(byte[] data) throws IOException {
        return unpack(data);
    }
    
    /**
     * Builds the payload bytes based on message type.
     */
    private byte[] buildPayload() throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(baos);
        
        // Write common fields
        writeString(dos, workerId != null ? workerId : "");
        dos.writeLong(timestamp);
        
        // Write type-specific fields
        switch (messageTypeEnum) {
            case REGISTER:
                // No additional fields needed
                break;
                
            case TASK_ASSIGNMENT:
                dos.writeLong(taskId);
                dos.writeInt(startRow);
                dos.writeInt(endRow);
                writeMatrix(dos, matrixA);
                writeMatrix(dos, matrixB);
                break;
                
            case TASK_RESULT:
                dos.writeLong(taskId);
                writeMatrix(dos, resultMatrix);
                break;
                
            case HEARTBEAT:
                // No additional fields
                break;
                
            case SHUTDOWN:
            case ACK:
            case ERROR:
                // Minimal payload
                break;
        }
        
        dos.flush();
        return baos.toByteArray();
    }
    
    /**
     * Parses payload bytes based on message type.
     */
    private void parsePayload(byte[] payloadBytes) throws IOException {
        if (payloadBytes.length == 0) return;
        
        DataInputStream dis = new DataInputStream(new ByteArrayInputStream(payloadBytes));
        
        // Read common fields
        this.workerId = readString(dis);
        this.timestamp = dis.readLong();
        
        // Read type-specific fields
        switch (messageTypeEnum) {
            case REGISTER:
                // No additional fields
                break;
                
            case TASK_ASSIGNMENT:
                this.taskId = dis.readLong();
                this.startRow = dis.readInt();
                this.endRow = dis.readInt();
                this.matrixA = readMatrix(dis);
                this.matrixB = readMatrix(dis);
                break;
                
            case TASK_RESULT:
                this.taskId = dis.readLong();
                this.resultMatrix = readMatrix(dis);
                break;
                
            case HEARTBEAT:
            case SHUTDOWN:
            case ACK:
            case ERROR:
                // No additional fields
                break;
        }
    }
    
    /**
     * Serializes a matrix to output stream.
     * Format: [rows:4][cols:4][row0_data...][row1_data...]...
     */
    private void writeMatrix(DataOutputStream dos, int[][] matrix) throws IOException {
        if (matrix == null || matrix.length == 0) {
            dos.writeInt(0); // rows
            dos.writeInt(0); // cols
            return;
        }
        
        int rows = matrix.length;
        int cols = matrix[0].length;
        
        dos.writeInt(rows);
        dos.writeInt(cols);
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                dos.writeInt(matrix[i][j]);
            }
        }
    }
    
    /**
     * Deserializes a matrix from input stream.
     */
    private int[][] readMatrix(DataInputStream dis) throws IOException {
        int rows = dis.readInt();
        int cols = dis.readInt();
        
        if (rows == 0 || cols == 0) {
            return new int[0][0];
        }
        
        if (rows < 0 || cols < 0 || rows > 10000 || cols > 10000) {
            throw new ProtocolException("Invalid matrix dimensions: " + rows + "x" + cols);
        }
        
        int[][] matrix = new int[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = dis.readInt();
            }
        }
        
        return matrix;
    }
    
    /**
     * Writes a length-prefixed UTF-8 string.
     */
    private void writeString(DataOutputStream dos, String str) throws IOException {
        byte[] bytes = str.getBytes(StandardCharsets.UTF_8);
        dos.writeInt(bytes.length);
        dos.write(bytes);
    }
    
    /**
     * Reads a length-prefixed UTF-8 string.
     */
    private String readString(DataInputStream dis) throws IOException {
        int length = dis.readInt();
        if (length < 0 || length > 1024) { // Reasonable string length limit
            throw new ProtocolException("Invalid string length: " + length);
        }
        if (length == 0) return "";
        
        byte[] bytes = new byte[length];
        dis.readFully(bytes);
        return new String(bytes, StandardCharsets.UTF_8);
    }
    
    /**
     * Helper to read exactly N bytes from stream, blocking until complete.
     * Critical for handling TCP fragmentation.
     */
    public static byte[] readMessageFromStream(InputStream in) throws IOException {
        // Read header (12 bytes)
        byte[] header = new byte[HEADER_SIZE];
        readFully(in, header);
        
        ByteBuffer headerBuf = ByteBuffer.wrap(header);
        int magic = headerBuf.getInt();
        int payloadLength = headerBuf.getInt();
        
        if (magic != MAGIC_NUMBER) {
            throw new ProtocolException("Invalid magic: 0x" + Integer.toHexString(magic));
        }
        
        if (payloadLength < 0 || payloadLength > MAX_PAYLOAD_SIZE) {
            throw new ProtocolException("Invalid payload length: " + payloadLength);
        }
        
        // Read payload
        byte[] payload = new byte[payloadLength];
        if (payloadLength > 0) {
            readFully(in, payload);
        }
        
        // Combine header + payload for unpack()
        byte[] fullMessage = new byte[HEADER_SIZE + payloadLength];
        System.arraycopy(header, 0, fullMessage, 0, HEADER_SIZE);
        System.arraycopy(payload, 0, fullMessage, HEADER_SIZE, payloadLength);
        
        return fullMessage;
    }
    
    /**
     * Reads exactly buffer.length bytes from stream.
     * Handles TCP fragmentation by looping until complete.
     */
    private static void readFully(InputStream in, byte[] buffer) throws IOException {
        int offset = 0;
        while (offset < buffer.length) {
            int bytesRead = in.read(buffer, offset, buffer.length - offset);
            if (bytesRead == -1) {
                throw new EOFException("Stream closed before message complete");
            }
            offset += bytesRead;
        }
    }
    
    /**
     * Custom exception for protocol violations.
     */
    public static class ProtocolException extends IOException {
        public ProtocolException(String message) {
            super(message);
        }
    }
    
    @Override
    public String toString() {
        return String.format("Message{type=%s, workerId=%s, taskId=%d, timestamp=%d}", 
            messageTypeEnum, workerId, taskId, timestamp);
    }
}
