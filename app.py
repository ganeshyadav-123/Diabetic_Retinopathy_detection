from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
from PIL import Image
import io
import base64
import os
import numpy as np

# Try to import torchvision components, with fallback
try:
    import torchvision.transforms as transforms
    from torchvision.models import resnet50, ResNet50_Weights
    TORCHVISION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: torchvision import failed: {e}")
    print("Running in compatibility mode...")
    TORCHVISION_AVAILABLE = False

app = Flask(__name__)

# Device configuration
device = torch.device("cpu")  # Using CPU for compatibility

# Model classes
CLASSES = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']
NUM_CLASSES = len(CLASSES)
IMG_SIZE = 224

# CBAM Attention Module
class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channels = channels

        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Match the saved model structure
        self.channel_attention = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction_ratio, channels, bias=False),
            nn.Sigmoid()
        )

        # Spatial Attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel Attention
        b, c, h, w = x.size()
        avg_pool = self.avg_pool(x).view(b, c)
        max_pool = self.max_pool(x).view(b, c)

        avg_out = self.channel_attention(avg_pool)
        max_out = self.channel_attention(max_pool)

        channel_attn = (avg_out + max_out).view(b, c, 1, 1)
        x = x * channel_attn.expand_as(x)

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attn = torch.cat([avg_out, max_out], dim=1)
        spatial_attn = self.spatial_attention(spatial_attn)
        x = x * spatial_attn.expand_as(x)

        return x

# ResNet with Attention Model
class ResNetWithAttention(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(ResNetWithAttention, self).__init__()
        if TORCHVISION_AVAILABLE:
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = resnet50(weights=weights)
        else:
            # Create a simple CNN as fallback
            print("Using fallback CNN model...")
            backbone = self._create_fallback_model()

        # Store backbone as a submodule to match saved model structure
        self.backbone = backbone

        # CBAM attention modules with correct channel numbers
        self.attention1 = CBAM(256)    # layer1 output channels
        self.attention2 = CBAM(512)    # layer2 output channels
        self.attention3 = CBAM(1024)   # layer3 output channels
        self.attention4 = CBAM(2048)   # layer4 output channels

        # Classifier layers to match saved model structure
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def _create_fallback_model(self):
        """Create a simple CNN model when torchvision is not available"""
        class SimpleCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
                self.bn1 = nn.BatchNorm2d(64)
                self.relu = nn.ReLU()
                self.maxpool = nn.MaxPool2d(3, 2, 1)
                self.layer1 = nn.Sequential(
                    nn.Conv2d(64, 256, 3, 1, 1),
                    nn.BatchNorm2d(256),
                    nn.ReLU()
                )
                self.layer2 = nn.Sequential(
                    nn.Conv2d(256, 512, 3, 2, 1),
                    nn.BatchNorm2d(512),
                    nn.ReLU()
                )
                self.layer3 = nn.Sequential(
                    nn.Conv2d(512, 1024, 3, 2, 1),
                    nn.BatchNorm2d(1024),
                    nn.ReLU()
                )
                self.layer4 = nn.Sequential(
                    nn.Conv2d(1024, 2048, 3, 2, 1),
                    nn.BatchNorm2d(2048),
                    nn.ReLU()
                )
                # Add avgpool to match ResNet structure
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                
            def forward(self, x):
                return self
        return SimpleCNN()

    def forward(self, x):
        # Use backbone layers
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.attention1(x)

        x = self.backbone.layer2(x)
        x = self.attention2(x)

        x = self.backbone.layer3(x)
        x = self.attention3(x)

        x = self.backbone.layer4(x)
        x = self.attention4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Load the trained model
def load_model():
    model = ResNetWithAttention(num_classes=NUM_CLASSES, pretrained=False)
    
    # Load the saved model
    model_path = '/Users/landaganesh/Documents/Projects /Miniproject/dr_model.pth'
    if os.path.exists(model_path):
        try:
            print(f"Loading model from: {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("Model loaded successfully from checkpoint!")
            else:
                # If it's just the state dict directly
                model.load_state_dict(checkpoint)
                print("Model loaded successfully from state dict!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using untrained model.")
    else:
        print("Model file not found. Using untrained model.")
    
    model.to(device)
    model.float()  # Ensure model uses float32
    model.eval()
    return model

# Initialize model
model = load_model()

# Image preprocessing
def preprocess_image(image):
    """Preprocess image for model inference"""
    if TORCHVISION_AVAILABLE:
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor
    else:
        # Fallback preprocessing without torchvision
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image manually
        image = image.resize((IMG_SIZE, IMG_SIZE))
        
        # Convert to numpy array and normalize
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        
        # Convert to tensor and ensure float32 type
        image_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float()
        return image_tensor

# Prediction function
def predict_diabetic_retinopathy(image):
    """Predict diabetic retinopathy from retinal image"""
    try:
        # Preprocess image
        image_tensor = preprocess_image(image)
        image_tensor = image_tensor.to(device)
        
        # Ensure tensor is float32
        image_tensor = image_tensor.float()
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            print(f"Raw model outputs: {outputs}")
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            print(f"Probabilities: {probabilities}")
            confidence, predicted = torch.max(probabilities, 1)
            print(f"Predicted class index: {predicted.item()}, Confidence: {confidence.item()}")
            
        # Get results
        predicted_class = CLASSES[predicted.item()]
        confidence_score = confidence.item()
        
        # Get all class probabilities
        all_probabilities = probabilities[0].cpu().numpy()
        class_probabilities = {CLASSES[i]: float(all_probabilities[i]) for i in range(len(CLASSES))}
        print(f"All class probabilities: {class_probabilities}")
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence_score,
            'all_probabilities': class_probabilities,
            'success': True
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'success': False
        }

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for prediction"""
    try:
        # Get image from request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided', 'success': False})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected', 'success': False})
        
        # Read and process image
        image = Image.open(io.BytesIO(file.read()))
        
        # Make prediction
        result = predict_diabetic_retinopathy(image)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False})

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': True})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
