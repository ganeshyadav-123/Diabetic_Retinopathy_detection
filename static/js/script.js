// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const imageInput = document.getElementById('imageInput');
const imagePreview = document.getElementById('imagePreview');
const previewImage = document.getElementById('previewImage');
const removeImage = document.getElementById('removeImage');
const analyzeButton = document.getElementById('analyzeButton');
const resultsContainer = document.getElementById('resultsContainer');
const loadingModal = document.getElementById('loadingModal');

// Upload functionality
uploadArea.addEventListener('click', () => imageInput.click());
uploadArea.addEventListener('dragover', handleDragOver);
uploadArea.addEventListener('dragleave', handleDragLeave);
uploadArea.addEventListener('drop', handleDrop);
imageInput.addEventListener('change', handleFileSelect);

// Image preview functionality
removeImage.addEventListener('click', removePreview);
analyzeButton.addEventListener('click', analyzeImage);

// Smooth scrolling for navigation
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Mobile navigation toggle
const navToggle = document.querySelector('.nav-toggle');
const navMenu = document.querySelector('.nav-menu');

navToggle.addEventListener('click', () => {
    navMenu.classList.toggle('active');
});

// Drag and drop handlers
function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        alert('Please select a valid image file.');
        return;
    }
    
    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        alert('File size too large. Please select an image smaller than 10MB.');
        return;
    }
    
    const reader = new FileReader();
    reader.onload = function(e) {
        previewImage.src = e.target.result;
        imagePreview.style.display = 'block';
        uploadArea.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

function removePreview() {
    imagePreview.style.display = 'none';
    uploadArea.style.display = 'block';
    resultsContainer.style.display = 'none';
    imageInput.value = '';
}

async function analyzeImage() {
    const file = imageInput.files[0];
    if (!file) {
        alert('Please select an image first.');
        return;
    }
    
    // Show loading modal
    loadingModal.style.display = 'flex';
    
    try {
        const formData = new FormData();
        formData.append('image', file);
        
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            displayResults(result);
        } else {
            alert('Error analyzing image: ' + result.error);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while analyzing the image.');
    } finally {
        // Hide loading modal
        loadingModal.style.display = 'none';
    }
}

function displayResults(result) {
    const {
        predicted_class,
        confidence,
        all_probabilities
    } = result;
    
    // Update prediction
    document.getElementById('predictionValue').textContent = predicted_class;
    document.getElementById('confidenceBadge').textContent = `${(confidence * 100).toFixed(1)}% Confidence`;
    
    // Update prediction description
    const descriptions = {
        'No_DR': 'No signs of diabetic retinopathy detected. Continue regular eye examinations.',
        'Mild': 'Early signs of diabetic retinopathy detected. Monitor closely and maintain good diabetes control.',
        'Moderate': 'Moderate diabetic retinopathy progression. Consider consultation with an ophthalmologist.',
        'Severe': 'Severe diabetic retinopathy detected. Immediate consultation with an ophthalmologist recommended.',
        'Proliferate_DR': 'Advanced proliferative diabetic retinopathy detected. Urgent medical attention required.'
    };
    
    document.getElementById('predictionDescription').textContent = descriptions[predicted_class] || '';
    
    // Update probability bars
    const probabilityBars = document.getElementById('probabilityBars');
    probabilityBars.innerHTML = '';
    
    const severityLevels = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR'];
    const severityColors = {
        'No_DR': '#10b981',
        'Mild': '#f59e0b',
        'Moderate': '#f97316',
        'Severe': '#ef4444',
        'Proliferate_DR': '#dc2626'
    };
    
    severityLevels.forEach(level => {
        const probability = all_probabilities[level] || 0;
        const percentage = (probability * 100).toFixed(1);
        
        const barContainer = document.createElement('div');
        barContainer.className = 'probability-bar';
        
        barContainer.innerHTML = `
            <div class="probability-label">${level.replace('_', ' ')}</div>
            <div class="probability-bar-container">
                <div class="probability-bar-fill ${level.toLowerCase().replace('_', '-')}" 
                     style="width: ${percentage}%; background-color: ${severityColors[level]}"></div>
            </div>
            <div class="probability-value">${percentage}%</div>
        `;
        
        probabilityBars.appendChild(barContainer);
    });
    
    // Update recommendations
    const recommendations = {
        'No_DR': 'Continue regular eye examinations every 6-12 months. Maintain good diabetes control and healthy lifestyle.',
        'Mild': 'Schedule follow-up examination in 3-6 months. Focus on optimal diabetes management and blood pressure control.',
        'Moderate': 'Consult with an ophthalmologist within 1-2 months. Consider laser treatment or other interventions.',
        'Severe': 'Seek immediate consultation with a retinal specialist. Treatment may include laser therapy or injections.',
        'Proliferate_DR': 'Urgent referral to a retinal specialist required. Immediate treatment may be necessary to prevent vision loss.'
    };
    
    document.getElementById('recommendationContent').textContent = recommendations[predicted_class] || '';
    
    // Show results
    resultsContainer.style.display = 'block';
    
    // Scroll to results
    resultsContainer.scrollIntoView({ behavior: 'smooth' });
}

// Add scroll effect to navigation
window.addEventListener('scroll', () => {
    const navbar = document.querySelector('.navbar');
    if (window.scrollY > 100) {
        navbar.style.background = 'linear-gradient(135deg, rgba(30, 58, 138, 0.95) 0%, rgba(59, 130, 246, 0.95) 100%)';
        navbar.style.backdropFilter = 'blur(10px)';
    } else {
        navbar.style.background = 'linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%)';
        navbar.style.backdropFilter = 'none';
    }
});

// Add active class to navigation links based on scroll position
window.addEventListener('scroll', () => {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-link');
    
    let current = '';
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.clientHeight;
        if (window.scrollY >= (sectionTop - 200)) {
            current = section.getAttribute('id');
        }
    });
    
    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${current}`) {
            link.classList.add('active');
        }
    });
});

// Form submission handler
document.querySelector('.contact-form form').addEventListener('submit', function(e) {
    e.preventDefault();
    
    // Get form data
    const formData = new FormData(this);
    const name = this.querySelector('input[type="text"]').value;
    const email = this.querySelector('input[type="email"]').value;
    const message = this.querySelector('textarea').value;
    
    // Simple validation
    if (!name || !email || !message) {
        alert('Please fill in all fields.');
        return;
    }
    
    // Simulate form submission
    const submitButton = this.querySelector('.submit-button');
    const originalText = submitButton.innerHTML;
    
    submitButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Sending...';
    submitButton.disabled = true;
    
    setTimeout(() => {
        alert('Thank you for your message! We will get back to you soon.');
        this.reset();
        submitButton.innerHTML = originalText;
        submitButton.disabled = false;
    }, 2000);
});

// Add animation on scroll
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Observe elements for animation
document.querySelectorAll('.about-card, .severity-card, .contact-item').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(20px)';
    el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    observer.observe(el);
});

// Add loading states and error handling
function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #ef4444;
        color: white;
        padding: 1rem 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);
        z-index: 3000;
        animation: slideIn 0.3s ease;
    `;
    errorDiv.textContent = message;
    
    document.body.appendChild(errorDiv);
    
    setTimeout(() => {
        errorDiv.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => errorDiv.remove(), 300);
    }, 5000);
}

function showSuccess(message) {
    const successDiv = document.createElement('div');
    successDiv.className = 'success-message';
    successDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #10b981;
        color: white;
        padding: 1rem 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
        z-index: 3000;
        animation: slideIn 0.3s ease;
    `;
    successDiv.textContent = message;
    
    document.body.appendChild(successDiv);
    
    setTimeout(() => {
        successDiv.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => successDiv.remove(), 300);
    }, 3000);
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(style);
