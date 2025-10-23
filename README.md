# Diabetic Retinopathy Detection Web Application

A professional web application for AI-powered diabetic retinopathy detection using deep learning technology.

## Features

- **AI-Powered Detection**: ResNet50 with CBAM attention mechanism for accurate classification
- **5 Severity Levels**: No DR, Mild, Moderate, Severe, and Proliferative DR
- **Professional UI**: Medical-themed responsive design
- **Real-time Analysis**: Fast image processing and instant results
- **Comprehensive Results**: Detailed probability breakdown and medical recommendations

## Technology Stack

- **Backend**: Flask (Python)
- **AI Model**: PyTorch with ResNet50 + CBAM Attention
- **Frontend**: HTML5, CSS3, JavaScript
- **Image Processing**: PIL/Pillow
- **Styling**: Custom CSS with medical color scheme

## Installation

1. **Clone or download the project files**

2. **Install Python dependencies**:
   ```bash
   pip install -r web_requirements.txt
   ```

3. **Ensure your model file is present**:
   - Make sure `dr_model.pth` is in the project root directory
   - This should be your trained model from the Jupyter notebook

## Running the Application

1. **Start the Flask server**:
   ```bash
   python app.py
   ```

2. **Open your web browser** and navigate to:
   ```
   http://localhost:5000
   ```

3. **Upload a retinal image** and get instant AI-powered analysis

## Usage

1. **Upload Image**: Drag and drop or click to select a retinal fundus image
2. **Analyze**: Click "Analyze Image" to process with AI
3. **View Results**: Get detailed classification with confidence scores
4. **Review Recommendations**: See medical recommendations based on results

## Model Information

- **Architecture**: ResNet50 with CBAM (Convolutional Block Attention Module)
- **Training Data**: 3,662 retinal fundus images
- **Classes**: 5 severity levels of diabetic retinopathy
- **Input Size**: 224x224 pixels
- **Accuracy**: 95%+ on validation set

## File Structure

```
├── app.py                          # Flask application
├── dr_model.pth                    # Trained model weights
├── templates/
│   ├── index.html                  # Main page
│   └── about.html                  # About page
├── static/
│   ├── css/
│   │   └── style.css               # Styling
│   └── js/
│       └── script.js               # JavaScript functionality
├── web_requirements.txt            # Python dependencies
└── README.md                       # This file
```

## API Endpoints

- `GET /` - Main application page
- `GET /about` - About page
- `POST /predict` - Image analysis endpoint
- `GET /api/health` - Health check

## Important Notes

⚠️ **Medical Disclaimer**: This system is for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis, treatment, or advice. Always consult with qualified healthcare professionals for medical decisions.

## Browser Compatibility

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Troubleshooting

1. **Model not loading**: Ensure `dr_model.pth` is in the correct location
2. **Image upload issues**: Check file format (PNG, JPG, JPEG supported)
3. **Performance**: The application runs on CPU by default for compatibility

## Development

To modify or extend the application:

1. **Backend**: Edit `app.py` for API changes
2. **Frontend**: Modify files in `templates/` and `static/`
3. **Styling**: Update `static/css/style.css`
4. **Functionality**: Edit `static/js/script.js`

## License

This project is for educational and research purposes. Please ensure compliance with medical device regulations if used in clinical settings.

## Support

For technical support or questions, please refer to the contact information in the application.
