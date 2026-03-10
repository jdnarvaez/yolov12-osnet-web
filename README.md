# YOLOv12 nano + OSNet ONNX Runtime Web

![react](https://img.shields.io/badge/React-blue?logo=react)
![onnxruntime-web](https://img.shields.io/badge/onnxruntime--web-white?logo=onnx&logoColor=black)

A minimalistic real-time object detection application built with YOLOv12 nano, OSNet re-id and ONNX Runtime Web for browser-based AI inference.

Users select a camera input, that stream of that camera is transferred to a WebWorker, inference is then performed in the WebWorker via `onnxruntimeweb` in a series of `OffscreenCanvas` objects. Bounding boxes are detected via YOLOv12 nano, and then those boxes are fed into a ReIDEngine that extracts a subimage based on each bounding box and track IDs are generated for each box. 

### Prerequisites

- Node.js 25.8.0+ (referenced in `.nvmrc`) 
- Modern browser with WebGPU support (Chrome, Edge, or Firefox)
- Camera access

### Installation

1. **Clone and install dependencies:**
   ```bash
   git clone https://github.com/jdnarvaez/yolov12-osnet-web.git
   cd yolov12-onnxruntime-web
   npm install
   ```

1. **Start the development server:**
   ```bash
   npm run dev
   ```

1. **Open your browser:**
   Navigate to `http://localhost:4173`

### Build for Production

```bash
pnpm run build
```

The built files will be in the `dist` directory, ready to be deployed to GitHub Pages or any static hosting service.

### Frontend Stack
- **TypeScript**: Type safety and better development experience
- **Vite**: build tool and dev server
- **React 19**: UI framework and state management
- **heroUI v3**: component library
- **Tailwind CSS v4**: styling

### AI/ML Stack
- **ONNX Runtime Web**: Browser-based AI inference
- **YOLOv12n**: Object detection model architecture
- **OSNet ReID**: Attempt to unique re-identify objects within the bounding boxes generated via YOLOv12 pass
- **In Browser Processing**: No server required

- **Inspired by:** [emergentai / yolov12-onnxruntime-web](https://github.com/emergentai/yolov12-onnxruntime-web)  

### Exporting a model

#### setup python venv

```bash
python3 -m venv .venv
source .venv/bin/activate
./scripts/export_osnet.sh
```