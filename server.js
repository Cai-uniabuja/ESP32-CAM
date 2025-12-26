import express from "express";
import fs from "fs";
import path from "path";
import multer from "multer";
import cors from "cors";
import fetch from "node-fetch";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = 5000;

// =========================
// Middleware
// =========================
app.use(cors());
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true }));

// =========================
// Upload folder
// =========================
const uploadDir = path.join(__dirname, "uploads");
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir);

// =========================
// Multer setup
// =========================
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) => {
    cb(null, `frame_${Date.now()}.jpg`);
  }
});
const upload = multer({
  storage,
  fileFilter: (req, file, cb) => {
    if (!file.mimetype.startsWith("image/"))
      return cb(new Error("Only images allowed"), false);
    cb(null, true);
  }
});

// =========================
// Root route
// =========================
app.get("/", (req, res) => res.json({ message: "ESP32 Surveillance Server" }));

// =========================
// Image Upload (ESP32→Server)
// =========================
app.post("/upload", express.raw({ type: "image/jpeg", limit: "10mb" }), (req, res) => {
  try {
    const filename = `frame_${Date.now()}.jpg`;
    fs.writeFileSync(path.join(uploadDir, filename), req.body);

    console.log(`Saved frame: ${filename}`);
    res.status(200).json({ status: "ok", file: filename });

  } catch (err) {
    console.error("Upload error:", err);
    res.status(500).json({ error: "Upload failed" });
  }
});

// =========================
// MJPEG Stream (Python→Node)
// =========================
app.get("/stream", async (req, res) => {
  res.writeHead(200, {
    "Content-Type": "multipart/x-mixed-replace; boundary=frame",
    "Cache-Control": "no-cache",
    Connection: "close",
    Pragma: "no-cache"
  });

  const boundary = "--frame";

  async function getFrame() {
    try {
      const camUrl = "http://172.20.10.3/capture";
      const response = await fetch(camUrl);
      if (!response.ok) throw new Error("Camera failed");

      return Buffer.from(await response.arrayBuffer());

    } catch (err) {
      console.error("⚠ ESP32 fetch error:", err.message);
      return null;
    }
  }

  const interval = setInterval(async () => {
    const frame = await getFrame();
    if (!frame) return;

    res.write(`${boundary}\r\n`);
    res.write("Content-Type: image/jpeg\r\n");
    res.write(`Content-Length: ${frame.length}\r\n\r\n`);
    res.write(frame);
    res.write("\r\n");
  }, 200);

  req.on("close", () => clearInterval(interval));
});

// =========================
// Snapshot (Used by auto_capture.py)
// =========================
app.get("/snapshot", async (req, res) => {
  try {
    const camUrl = "http://172.20.10.3/capture";
    const result = await fetch(camUrl);

    if (!result.ok) throw new Error("Snapshot failed");

    const buffer = await result.arrayBuffer();

    res.set("Content-Type", "image/jpeg");
    res.send(Buffer.from(buffer));

  } catch (err) {
    console.error(" Snapshot Error:", err.message);
    res.status(500).send("Snapshot failed");
  }
});

// =========================
// Attendance POST Endpoint
// =========================
app.post("/attendance", (req, res) => {
  try {
    const attendance = req.body;
    console.log("Attendance:", attendance);

    // Placeholder for ConvexDB integration
    // await convex.mutation("attendance.insert", attendance);

    res.status(200).json({ success: true, received: attendance });

  } catch (err) {
    console.error(" Attendance error:", err);
    res.status(500).send("Attendance failed");
  }
});

// =========================
// Static files (Viewing uploads)
// =========================
app.use("/uploads", express.static(uploadDir));

// =========================
// Error handler
// =========================
app.use((err, req, res, next) => {
  console.error(" Global Error:", err.message);
  res.status(500).json({ error: err.message });
});

// =========================
// Start server
// =========================
app.listen(port, () => {
  console.log(` Server: http://localhost:${port}`);
  console.log(` Snapshot: http://localhost:${port}/snapshot`);
  console.log(` Stream: http://localhost:${port}/stream`);
  console.log(` Attendance: http://localhost:${port}/attendance`);
});

