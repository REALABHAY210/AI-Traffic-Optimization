AI-Based Traffic Optimization & Smart NavigationAn intelligent traffic monitoring and management system that leverages YOLOv8 for real-time vehicle detection and FastAPI to serve traffic density data. 
This project is designed to bridge the gap between computer vision and smart city infrastructure.
🚀 FeaturesReal-time Detection: High-speed vehicle tracking (cars, motorcycles, buses, trucks) using the YOLOv8 nano model.
Smart Counting: Specialized logic to filter and count specific vehicle types to calculate road density.
Dual-Interface API:GET /count: Returns a JSON object with the current vehicle count.GET /video_feed: Provides an MJPEG stream for real-time visual monitoring.
Containerized Deployment: Includes Docker and Docker Compose configurations for both training and API serving.
Frontend Integration: Next.js-ready structure for a modern dashboard UI.🛠️ Project StructurePlaintext.
├── frontend/             # Next.js Dashboard
├── yolo/                 # Computer Vision Core
│   ├── main.py           # FastAPI Server & Inference Loop
│   ├── cartest.py        # Local testing script for webcam
│   └── yolov8n.pt        # YOLOv8 Model Weights
├── docker/               # Deployment Configs
│   ├── dockerfile.api    # API Server Environment
│   └── dockerfile.train  # ML Training Environment
└── README.md
⚡ Quick Start1. PrerequisitesPython 3.10+OpenCV dependenciesWebcam (for local testing)2. Local InstallationBash# Clone the repository
git clone https://github.com/REALABHAY210/AI_based_traffixc.git
cd AI_based_traffixc

# Create virtual environment
python -m venv venv
source venv/bin/scripts/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install ultralytics fastapi uvicorn opencv-python
3. Running the APIBashpython yolo/main.py
The server will start at http://localhost:8000. You can view the live stream at http://localhost:8000/video_feed.
🐳 Docker DeploymentTo run the API in a containerized environment with GPU support:Bashdocker compose -f docker-compose.api.yaml up --build
📊 EndpointsEndpointMethodDescription/countGETReturns current vehicle density as JSON./video_feedGETLive MJPEG stream of processed frames.
/frameGETCaptures a single processed JPEG snapshot.📝 LicenseDistributed under the MIT License. 
See LICENSE for more information.
🤝 ContactAbhay Partap Singh Project Link: https://github.com/REALABHAY210/AI_based_traffixc
