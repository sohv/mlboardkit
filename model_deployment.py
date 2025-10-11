#!/usr/bin/env python3
"""
Model Deployment and Serving Utilities

Comprehensive toolkit for ML model deployment including model packaging, API creation,
containerization, and serving infrastructure setup.

Usage:
    python3 model_deployment.py package --model model.pkl --output model_package/
    python3 model_deployment.py serve --model model_package/ --port 8000
    python3 model_deployment.py docker --model model_package/ --output Dockerfile
    python3 model_deployment.py api --model model_package/ --framework flask
    python3 model_deployment.py test --endpoint http://localhost:8000/predict --data test.json
"""

import argparse
import json
import pickle
import joblib
import os
import sys
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import logging
import yaml
from dataclasses import dataclass, asdict
import hashlib
import zipfile
import base64
import requests


@dataclass
class ModelMetadata:
    """Model metadata container."""
    name: str
    version: str
    framework: str
    model_type: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    preprocessing: Optional[Dict[str, Any]] = None
    postprocessing: Optional[Dict[str, Any]] = None
    dependencies: Optional[List[str]] = None
    created_at: str = ""
    description: str = ""
    author: str = ""
    tags: Optional[List[str]] = None
    performance_metrics: Optional[Dict[str, float]] = None
    model_size_mb: Optional[float] = None
    inference_time_ms: Optional[float] = None


class ModelPackager:
    """Model packaging utilities."""
    
    def __init__(self):
        self.supported_formats = ['pickle', 'joblib', 'pytorch', 'tensorflow', 'onnx']
    
    def package_model(self, model_path: str, output_dir: str, 
                     metadata: ModelMetadata) -> Path:
        """Package model with metadata and dependencies."""
        model_path = Path(model_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📦 Packaging model: {model_path}")
        
        # Detect model format
        model_format = self._detect_model_format(model_path)
        metadata.framework = model_format
        
        # Calculate model size
        metadata.model_size_mb = model_path.stat().st_size / (1024 * 1024)
        metadata.created_at = datetime.now().isoformat()
        
        # Copy model file
        model_dest = output_dir / f"model.{model_format}"
        if model_path.is_file():
            shutil.copy2(model_path, model_dest)
        else:
            shutil.copytree(model_path, model_dest)
        
        # Create model info
        model_info = {
            'metadata': asdict(metadata),
            'files': {
                'model': f"model.{model_format}",
                'metadata': 'metadata.json',
                'requirements': 'requirements.txt',
                'inference_script': 'inference.py',
                'api_script': 'api.py'
            },
            'checksum': self._calculate_checksum(model_dest)
        }
        
        # Save metadata
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(model_info, f, indent=2, default=str)
        
        # Create requirements.txt
        self._create_requirements_file(output_dir, metadata.dependencies)
        
        # Create inference script
        self._create_inference_script(output_dir, model_format, metadata)
        
        # Create API script
        self._create_api_script(output_dir, metadata)
        
        # Create Docker files
        self._create_dockerfile(output_dir, metadata)
        
        # Create deployment scripts
        self._create_deployment_scripts(output_dir, metadata)
        
        print(f"✅ Model packaged successfully: {output_dir}")
        return output_dir
    
    def _detect_model_format(self, model_path: Path) -> str:
        """Detect model format from file extension or content."""
        if model_path.is_file():
            suffix = model_path.suffix.lower()
            if suffix in ['.pkl', '.pickle']:
                return 'pickle'
            elif suffix == '.joblib':
                return 'joblib'
            elif suffix in ['.pt', '.pth']:
                return 'pytorch'
            elif suffix in ['.h5', '.keras']:
                return 'tensorflow'
            elif suffix == '.onnx':
                return 'onnx'
        
        # Check directory structure for framework-specific patterns
        if model_path.is_dir():
            if (model_path / 'saved_model.pb').exists():
                return 'tensorflow'
            elif any(f.suffix in ['.pt', '.pth'] for f in model_path.iterdir()):
                return 'pytorch'
        
        return 'unknown'
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file or directory."""
        sha256_hash = hashlib.sha256()
        
        if file_path.is_file():
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
        else:
            # For directories, hash all files
            for file in sorted(file_path.rglob('*')):
                if file.is_file():
                    with open(file, "rb") as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    def _create_requirements_file(self, output_dir: Path, dependencies: Optional[List[str]]):
        """Create requirements.txt file."""
        requirements = dependencies or []
        
        # Add common ML dependencies based on usage
        common_deps = ['numpy', 'pandas', 'scikit-learn']
        for dep in common_deps:
            if dep not in requirements:
                requirements.append(dep)
        
        with open(output_dir / 'requirements.txt', 'w') as f:
            for req in sorted(requirements):
                f.write(f"{req}\n")
    
    def _create_inference_script(self, output_dir: Path, model_format: str, metadata: ModelMetadata):
        """Create inference script for the model."""
        
        if model_format == 'pickle':
            load_code = """
import pickle
with open('model.pickle', 'rb') as f:
    model = pickle.load(f)
"""
            predict_code = "predictions = model.predict(input_data)"
        
        elif model_format == 'joblib':
            load_code = """
import joblib
model = joblib.load('model.joblib')
"""
            predict_code = "predictions = model.predict(input_data)"
        
        elif model_format == 'tensorflow':
            load_code = """
import tensorflow as tf
model = tf.keras.models.load_model('model.tensorflow')
"""
            predict_code = "predictions = model.predict(input_data)"
        
        elif model_format == 'pytorch':
            load_code = """
import torch
model = torch.load('model.pytorch')
model.eval()
"""
            predict_code = """
with torch.no_grad():
    predictions = model(input_data)
"""
        
        else:
            load_code = "# Model loading code for " + model_format
            predict_code = "# Prediction code"
        
        inference_script = f'''#!/usr/bin/env python3
"""
Inference script for {metadata.name}
Generated automatically by model deployment utilities.
"""

import numpy as np
import pandas as pd
import json
from typing import Any, Dict, List, Union

class ModelInference:
    """Model inference wrapper."""
    
    def __init__(self, model_path: str = '.'):
        """Initialize inference engine."""
        self.model_path = model_path
        self.model = None
        self.metadata = None
        self._load_model()
        self._load_metadata()
    
    def _load_model(self):
        """Load the trained model."""
        try:
{load_code}
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {{e}}")
            raise
    
    def _load_metadata(self):
        """Load model metadata."""
        try:
            with open('metadata.json', 'r') as f:
                self.metadata = json.load(f)
        except Exception as e:
            print(f"⚠️  Could not load metadata: {{e}}")
    
    def preprocess(self, data: Any) -> Any:
        """Preprocess input data."""
        # Add preprocessing logic based on metadata
        return data
    
    def predict(self, input_data: Any) -> Any:
        """Make predictions on input data."""
        try:
            # Preprocess
            processed_data = self.preprocess(input_data)
            
            # Predict
            {predict_code}
            
            # Postprocess
            result = self.postprocess(predictions)
            
            return result
            
        except Exception as e:
            print(f"❌ Prediction error: {{e}}")
            raise
    
    def postprocess(self, predictions: Any) -> Any:
        """Postprocess model predictions."""
        # Convert to serializable format
        if hasattr(predictions, 'numpy'):
            predictions = predictions.numpy()
        
        if hasattr(predictions, 'tolist'):
            return predictions.tolist()
        
        return predictions
    
    def predict_batch(self, batch_data: List[Any]) -> List[Any]:
        """Make predictions on batch of data."""
        results = []
        for data in batch_data:
            result = self.predict(data)
            results.append(result)
        return results


def main():
    """Command line interface for inference."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model inference")
    parser.add_argument('--input', required=True, help='Input data file (JSON)')
    parser.add_argument('--output', help='Output file for predictions')
    parser.add_argument('--batch', action='store_true', help='Process as batch')
    
    args = parser.parse_args()
    
    # Load inference engine
    inference = ModelInference()
    
    # Load input data
    with open(args.input, 'r') as f:
        input_data = json.load(f)
    
    # Make predictions
    if args.batch and isinstance(input_data, list):
        predictions = inference.predict_batch(input_data)
    else:
        predictions = inference.predict(input_data)
    
    # Save or print results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"✅ Predictions saved to: {{args.output}}")
    else:
        print(json.dumps(predictions, indent=2))


if __name__ == '__main__':
    main()
'''
        
        with open(output_dir / 'inference.py', 'w') as f:
            f.write(inference_script)
    
    def _create_api_script(self, output_dir: Path, metadata: ModelMetadata):
        """Create API server script."""
        
        api_script = f'''#!/usr/bin/env python3
"""
REST API server for {metadata.name}
Generated automatically by model deployment utilities.
"""

from flask import Flask, request, jsonify, render_template_string
from inference import ModelInference
import json
import traceback
from datetime import datetime
import os

app = Flask(__name__)

# Initialize model
try:
    model_inference = ModelInference()
    print("✅ Model loaded for API serving")
except Exception as e:
    print(f"❌ Failed to load model: {{e}}")
    model_inference = None

# HTML template for simple web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>{metadata.name} - Model API</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .container {{ max-width: 800px; margin: 0 auto; }}
        .form-group {{ margin: 20px 0; }}
        label {{ display: block; margin-bottom: 5px; font-weight: bold; }}
        textarea {{ width: 100%; height: 100px; padding: 10px; }}
        button {{ background: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; }}
        .result {{ margin-top: 20px; padding: 15px; background: #f8f9fa; border-left: 4px solid #007bff; }}
        .error {{ border-left-color: #dc3545; background: #f8d7da; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{metadata.name}</h1>
        <p><strong>Version:</strong> {metadata.version}</p>
        <p><strong>Framework:</strong> {metadata.framework}</p>
        <p><strong>Description:</strong> {metadata.description}</p>
        
        <h2>Test the Model</h2>
        <form id="predictionForm">
            <div class="form-group">
                <label for="inputData">Input Data (JSON format):</label>
                <textarea id="inputData" name="inputData" placeholder='Enter JSON data here...'></textarea>
            </div>
            <button type="submit">Make Prediction</button>
        </form>
        
        <div id="result" class="result" style="display: none;"></div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {{
            e.preventDefault();
            
            const inputData = document.getElementById('inputData').value;
            const resultDiv = document.getElementById('result');
            
            try {{
                const response = await fetch('/predict', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                    }},
                    body: inputData
                }});
                
                const result = await response.json();
                resultDiv.className = 'result';
                resultDiv.innerHTML = '<h3>Result:</h3><pre>' + JSON.stringify(result, null, 2) + '</pre>';
                resultDiv.style.display = 'block';
                
            }} catch (error) {{
                resultDiv.className = 'result error';
                resultDiv.innerHTML = '<h3>Error:</h3><p>' + error.message + '</p>';
                resultDiv.style.display = 'block';
            }}
        }});
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Serve simple web interface."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({{
        'status': 'healthy' if model_inference else 'unhealthy',
        'timestamp': datetime.now().isoformat(),
        'model': '{metadata.name}' if model_inference else None
    }})

@app.route('/info')
def info():
    """Model information endpoint."""
    if not model_inference:
        return jsonify({{'error': 'Model not loaded'}}), 500
    
    return jsonify({{
        'model_name': '{metadata.name}',
        'version': '{metadata.version}',
        'framework': '{metadata.framework}',
        'metadata': model_inference.metadata if model_inference.metadata else None
    }})

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint."""
    if not model_inference:
        return jsonify({{'error': 'Model not loaded'}}), 500
    
    try:
        # Get input data
        input_data = request.get_json()
        
        if input_data is None:
            return jsonify({{'error': 'No JSON data provided'}}), 400
        
        # Make prediction
        prediction = model_inference.predict(input_data)
        
        return jsonify({{
            'prediction': prediction,
            'timestamp': datetime.now().isoformat(),
            'model': '{metadata.name}'
        }})
        
    except Exception as e:
        error_msg = str(e)
        traceback.print_exc()
        return jsonify({{'error': error_msg}}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint."""
    if not model_inference:
        return jsonify({{'error': 'Model not loaded'}}), 500
    
    try:
        # Get input data
        input_data = request.get_json()
        
        if not isinstance(input_data, list):
            return jsonify({{'error': 'Batch data must be a list'}}), 400
        
        # Make predictions
        predictions = model_inference.predict_batch(input_data)
        
        return jsonify({{
            'predictions': predictions,
            'count': len(predictions),
            'timestamp': datetime.now().isoformat(),
            'model': '{metadata.name}'
        }})
        
    except Exception as e:
        error_msg = str(e)
        traceback.print_exc()
        return jsonify({{'error': error_msg}}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    print(f"🚀 Starting API server for {metadata.name}")
    print(f"📡 Server will be available at: http://localhost:{{port}}")
    print(f"📊 Web interface: http://localhost:{{port}}/")
    print(f"🔍 Health check: http://localhost:{{port}}/health")
    print(f"📈 Predictions: http://localhost:{{port}}/predict")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
'''
        
        with open(output_dir / 'api.py', 'w') as f:
            f.write(api_script)
    
    def _create_dockerfile(self, output_dir: Path, metadata: ModelMetadata):
        """Create Dockerfile for containerization."""
        
        dockerfile_content = f'''# Dockerfile for {metadata.name}
# Generated automatically by model deployment utilities

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "api.py"]
'''
        
        with open(output_dir / 'Dockerfile', 'w') as f:
            f.write(dockerfile_content)
        
        # Create .dockerignore
        dockerignore_content = '''# Docker ignore file
.git
.gitignore
README.md
.pytest_cache
.coverage
.env
*.log
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
'''
        
        with open(output_dir / '.dockerignore', 'w') as f:
            f.write(dockerignore_content)
    
    def _create_deployment_scripts(self, output_dir: Path, metadata: ModelMetadata):
        """Create deployment helper scripts."""
        
        # Docker build and run script
        docker_script = f'''#!/bin/bash
# Docker deployment script for {metadata.name}

set -e

IMAGE_NAME="{metadata.name.lower().replace(' ', '-')}"
IMAGE_TAG="{metadata.version}"
CONTAINER_NAME="${{IMAGE_NAME}}-container"
PORT=8000

echo "🐳 Building Docker image..."
docker build -t ${{IMAGE_NAME}}:${{IMAGE_TAG}} .
docker tag ${{IMAGE_NAME}}:${{IMAGE_TAG}} ${{IMAGE_NAME}}:latest

echo "🚀 Running container..."
docker stop ${{CONTAINER_NAME}} 2>/dev/null || true
docker rm ${{CONTAINER_NAME}} 2>/dev/null || true

docker run -d \\
    --name ${{CONTAINER_NAME}} \\
    -p ${{PORT}}:8000 \\
    -e DEBUG=false \\
    ${{IMAGE_NAME}}:${{IMAGE_TAG}}

echo "✅ Container started successfully!"
echo "📡 API available at: http://localhost:${{PORT}}"
echo "🔍 Health check: http://localhost:${{PORT}}/health"

# Show logs
echo "📋 Container logs:"
docker logs ${{CONTAINER_NAME}}
'''
        
        with open(output_dir / 'deploy_docker.sh', 'w') as f:
            f.write(docker_script)
        
        # Make executable
        os.chmod(output_dir / 'deploy_docker.sh', 0o755)
        
        # Local deployment script
        local_script = f'''#!/bin/bash
# Local deployment script for {metadata.name}

set -e

echo "🔧 Setting up local environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo "🚀 Starting local server..."
export PORT=8000
export DEBUG=true
python api.py
'''
        
        with open(output_dir / 'deploy_local.sh', 'w') as f:
            f.write(local_script)
        
        # Make executable
        os.chmod(output_dir / 'deploy_local.sh', 0o755)
        
        # Kubernetes deployment (basic)
        k8s_deployment = f'''apiVersion: apps/v1
kind: Deployment
metadata:
  name: {metadata.name.lower().replace(' ', '-')}-deployment
  labels:
    app: {metadata.name.lower().replace(' ', '-')}
spec:
  replicas: 2
  selector:
    matchLabels:
      app: {metadata.name.lower().replace(' ', '-')}
  template:
    metadata:
      labels:
        app: {metadata.name.lower().replace(' ', '-')}
    spec:
      containers:
      - name: model-api
        image: {metadata.name.lower().replace(' ', '-')}:{metadata.version}
        ports:
        - containerPort: 8000
        env:
        - name: DEBUG
          value: "false"
        resources:
          limits:
            memory: "1Gi"
            cpu: "500m"
          requests:
            memory: "512Mi"
            cpu: "250m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: {metadata.name.lower().replace(' ', '-')}-service
spec:
  selector:
    app: {metadata.name.lower().replace(' ', '-')}
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
'''
        
        with open(output_dir / 'k8s-deployment.yaml', 'w') as f:
            f.write(k8s_deployment)


class ModelServer:
    """Model serving utilities."""
    
    def __init__(self, model_package_path: str):
        self.package_path = Path(model_package_path)
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load model metadata."""
        metadata_path = self.package_path / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}
    
    def start_server(self, port: int = 8000, debug: bool = False):
        """Start the model API server."""
        api_script = self.package_path / 'api.py'
        if not api_script.exists():
            raise FileNotFoundError("API script not found in model package")
        
        print(f"🚀 Starting model server on port {port}")
        
        # Set environment variables
        env = os.environ.copy()
        env['PORT'] = str(port)
        env['DEBUG'] = str(debug).lower()
        
        # Change to package directory and run
        try:
            subprocess.run([
                sys.executable, 'api.py'
            ], cwd=self.package_path, env=env, check=True)
        except KeyboardInterrupt:
            print("\n🛑 Server stopped by user")
        except subprocess.CalledProcessError as e:
            print(f"❌ Server failed to start: {e}")
    
    def build_docker_image(self, image_name: Optional[str] = None, 
                          tag: str = 'latest') -> str:
        """Build Docker image for the model."""
        if not image_name:
            model_name = self.metadata.get('metadata', {}).get('name', 'model')
            image_name = model_name.lower().replace(' ', '-')
        
        full_image_name = f"{image_name}:{tag}"
        
        print(f"🐳 Building Docker image: {full_image_name}")
        
        try:
            result = subprocess.run([
                'docker', 'build', '-t', full_image_name, '.'
            ], cwd=self.package_path, check=True, capture_output=True, text=True)
            
            print(f"✅ Docker image built successfully: {full_image_name}")
            return full_image_name
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Docker build failed: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            raise
    
    def run_docker_container(self, image_name: str, port: int = 8000,
                           container_name: Optional[str] = None) -> str:
        """Run Docker container."""
        if not container_name:
            container_name = f"{image_name.split(':')[0]}-container"
        
        print(f"🚀 Running Docker container: {container_name}")
        
        try:
            # Stop existing container if running
            subprocess.run(['docker', 'stop', container_name], 
                         capture_output=True, text=True)
            subprocess.run(['docker', 'rm', container_name], 
                         capture_output=True, text=True)
            
            # Run new container
            result = subprocess.run([
                'docker', 'run', '-d',
                '--name', container_name,
                '-p', f"{port}:8000",
                image_name
            ], check=True, capture_output=True, text=True)
            
            container_id = result.stdout.strip()
            print(f"✅ Container started: {container_id[:12]}")
            print(f"📡 API available at: http://localhost:{port}")
            
            return container_id
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to run container: {e}")
            print(f"STDERR: {e.stderr}")
            raise


class ModelTester:
    """Model testing utilities."""
    
    def __init__(self):
        pass
    
    def test_endpoint(self, endpoint_url: str, test_data: Dict[str, Any],
                     expected_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """Test model endpoint with sample data."""
        print(f"🧪 Testing endpoint: {endpoint_url}")
        
        try:
            response = requests.post(
                endpoint_url,
                json=test_data,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Check response structure
            if expected_keys:
                missing_keys = [key for key in expected_keys if key not in result]
                if missing_keys:
                    print(f"⚠️  Missing expected keys: {missing_keys}")
            
            print(f"✅ Test successful - Status: {response.status_code}")
            return {
                'success': True,
                'status_code': response.status_code,
                'response': result,
                'response_time_ms': response.elapsed.total_seconds() * 1000
            }
            
        except requests.RequestException as e:
            print(f"❌ Test failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'response_time_ms': None
            }
    
    def load_test(self, endpoint_url: str, test_data: Dict[str, Any],
                 num_requests: int = 10, concurrent: bool = False) -> Dict[str, Any]:
        """Perform load testing on endpoint."""
        print(f"📊 Load testing endpoint with {num_requests} requests")
        
        results = []
        
        if concurrent:
            import concurrent.futures
            import threading
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(self.test_endpoint, endpoint_url, test_data)
                    for _ in range(num_requests)
                ]
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        results.append({'success': False, 'error': str(e)})
        else:
            for i in range(num_requests):
                result = self.test_endpoint(endpoint_url, test_data)
                results.append(result)
                if (i + 1) % 10 == 0:
                    print(f"📈 Completed {i + 1}/{num_requests} requests")
        
        # Analyze results
        successful_requests = [r for r in results if r.get('success')]
        failed_requests = [r for r in results if not r.get('success')]
        
        response_times = [r['response_time_ms'] for r in successful_requests 
                         if r.get('response_time_ms') is not None]
        
        analysis = {
            'total_requests': num_requests,
            'successful_requests': len(successful_requests),
            'failed_requests': len(failed_requests),
            'success_rate': len(successful_requests) / num_requests * 100,
            'avg_response_time_ms': sum(response_times) / len(response_times) if response_times else 0,
            'min_response_time_ms': min(response_times) if response_times else 0,
            'max_response_time_ms': max(response_times) if response_times else 0,
            'errors': [r.get('error') for r in failed_requests]
        }
        
        print(f"📊 Load test completed:")
        print(f"   Success rate: {analysis['success_rate']:.1f}%")
        print(f"   Avg response time: {analysis['avg_response_time_ms']:.1f}ms")
        
        return analysis


def main():
    parser = argparse.ArgumentParser(description="Model deployment and serving utilities")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Package command
    package_parser = subparsers.add_parser('package', help='Package model for deployment')
    package_parser.add_argument('--model', required=True, help='Model file or directory path')
    package_parser.add_argument('--output', required=True, help='Output directory for package')
    package_parser.add_argument('--name', required=True, help='Model name')
    package_parser.add_argument('--version', default='1.0.0', help='Model version')
    package_parser.add_argument('--description', default='', help='Model description')
    package_parser.add_argument('--framework', help='ML framework (auto-detected if not specified)')
    package_parser.add_argument('--dependencies', nargs='*', help='Additional dependencies')

    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start model API server')
    serve_parser.add_argument('--model', required=True, help='Model package directory')
    serve_parser.add_argument('--port', type=int, default=8000, help='Server port')
    serve_parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    # Docker commands
    docker_parser = subparsers.add_parser('docker', help='Docker operations')
    docker_parser.add_argument('--model', required=True, help='Model package directory')
    docker_parser.add_argument('--action', choices=['build', 'run', 'both'], 
                              default='both', help='Docker action')
    docker_parser.add_argument('--image-name', help='Docker image name')
    docker_parser.add_argument('--tag', default='latest', help='Docker image tag')
    docker_parser.add_argument('--port', type=int, default=8000, help='Container port mapping')

    # Test command
    test_parser = subparsers.add_parser('test', help='Test model endpoint')
    test_parser.add_argument('--endpoint', required=True, help='API endpoint URL')
    test_parser.add_argument('--data', required=True, help='Test data file (JSON)')
    test_parser.add_argument('--load-test', action='store_true', help='Perform load testing')
    test_parser.add_argument('--num-requests', type=int, default=10, help='Number of requests for load test')
    test_parser.add_argument('--concurrent', action='store_true', help='Use concurrent requests')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == 'package':
            # Create model metadata
            metadata = ModelMetadata(
                name=args.name,
                version=args.version,
                framework=args.framework or 'unknown',
                model_type='unknown',
                input_schema={},
                output_schema={},
                dependencies=args.dependencies,
                description=args.description
            )
            
            packager = ModelPackager()
            package_path = packager.package_model(args.model, args.output, metadata)
            
            print(f"📦 Model package created: {package_path}")
            print("🚀 To serve the model:")
            print(f"   python3 {__file__} serve --model {package_path}")
            print("🐳 To build Docker image:")
            print(f"   python3 {__file__} docker --model {package_path}")

        elif args.command == 'serve':
            server = ModelServer(args.model)
            server.start_server(args.port, args.debug)

        elif args.command == 'docker':
            server = ModelServer(args.model)
            
            if args.action in ['build', 'both']:
                image_name = server.build_docker_image(args.image_name, args.tag)
                
                if args.action == 'both':
                    server.run_docker_container(image_name, args.port)

        elif args.command == 'test':
            # Load test data
            with open(args.data, 'r') as f:
                test_data = json.load(f)
            
            tester = ModelTester()
            
            if args.load_test:
                results = tester.load_test(
                    args.endpoint, test_data, 
                    args.num_requests, args.concurrent
                )
                print(json.dumps(results, indent=2))
            else:
                result = tester.test_endpoint(args.endpoint, test_data)
                print(json.dumps(result, indent=2))

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()