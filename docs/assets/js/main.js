// Tab functionality
function switchTab(tabName) {
    // Hide all tab contents
    const tabContents = document.querySelectorAll('.tab-content');
    tabContents.forEach(content => {
        content.classList.remove('active');
    });
    
    // Remove active class from all tab buttons
    const tabButtons = document.querySelectorAll('.tab-button');
    tabButtons.forEach(button => {
        button.classList.remove('active');
    });
    
    // Show selected tab content
    const selectedTab = document.getElementById(tabName + '-tab');
    const selectedContent = document.getElementById(tabName + '-content');
    
    if (selectedTab && selectedContent) {
        selectedTab.classList.add('active');
        selectedContent.classList.add('active');
    }
}

// Feature data for all features
const featureData = {
    409: { activation: 1.4363, instances: [9, 4, 8, 5, 1, 7, 2, 6, 3, 0] },
    941: { activation: 1.2172, instances: [0, 2, 4, 8, 7, 5, 9, 1, 3, 6] },
    894: { activation: 1.1563, instances: [8, 3, 5, 0, 7, 6, 2, 9, 1, 4] },
    227: { activation: 1.1246, instances: [9, 1, 4, 8, 7, 5, 6, 2, 3, 0] },
    945: { activation: 1.1066, instances: [6, 7, 2, 3, 0, 1, 4, 9, 8, 5] },
    1022: { activation: 1.0379, instances: [7, 6, 2, 3, 0, 8, 4, 5, 9, 1] },
    348: { activation: 1.0149, instances: [6, 2, 0, 3, 8, 1, 5, 9, 7, 4] },
    874: { activation: 1.0119, instances: [6, 2, 7, 0, 3, 1, 4, 5, 8, 9] },
    762: { activation: 0.9816, instances: [1, 9, 6, 2, 4, 0, 3, 5, 8, 7] },
    764: { activation: 0.9442, instances: [6, 3, 2, 7, 9, 0, 4, 1, 8, 5] },
    239: { activation: 0.9366, instances: [1, 5, 8, 9, 4, 0, 2, 3, 6, 7] },
    951: { activation: 0.9297, instances: [7, 2, 6, 0, 3, 8, 5, 4, 9, 1] },
    374: { activation: 0.8736, instances: [1, 5, 9, 2, 6, 3, 4, 8, 0, 7] },
    898: { activation: 0.8448, instances: [7, 2, 8, 5, 6, 3, 0, 9, 4, 1] },
    50: { activation: 0.7647, instances: [0, 4, 2, 9, 8, 6, 7, 1, 3, 5] },
    323: { activation: 0.7569, instances: [6, 1, 2, 9, 3, 4, 0, 7, 5, 8] },
    82: { activation: 0.7413, instances: [9, 1, 8, 0, 4, 2, 5, 6, 3, 7] },
    948: { activation: 0.7388, instances: [7, 2, 6, 3, 8, 0, 9, 5, 4, 1] },
    978: { activation: 0.6830, instances: [9, 7, 8, 4, 3, 2, 5, 6, 1, 0] },
    272: { activation: 0.6765, instances: [1, 6, 3, 9, 0, 2, 4, 7, 5, 8] },
    770: { activation: 0.6470, instances: [6, 1, 2, 0, 3, 7, 9, 4, 8, 5] },
    555: { activation: 0.6164, instances: [1, 6, 9, 0, 4, 2, 3, 8, 7, 5] },
    730: { activation: 0.5839, instances: [1, 6, 0, 9, 4, 3, 2, 8, 7, 5] },
    329: { activation: 0.5732, instances: [1, 0, 4, 6, 9, 7, 2, 3, 5, 8] },
    689: { activation: 0.5724, instances: [6, 1, 2, 3, 9, 5, 7, 0, 4, 8] },
    287: { activation: 0.5704, instances: [6, 1, 7, 9, 4, 8, 3, 5, 2, 0] },
    310: { activation: 0.5394, instances: [7, 8, 0, 4, 9, 6, 5, 3, 1, 2] },
    243: { activation: 0.5304, instances: [0, 8, 2, 6, 5, 3, 1, 7, 4, 9] },
    710: { activation: 0.5256, instances: [1, 9, 8, 3, 5, 6, 7, 2, 4, 0] },
    63: { activation: 0.5132, instances: [8, 2, 9, 4, 7, 5, 3, 6, 0, 1] },
    596: { activation: 0.5053, instances: [1, 6, 2, 9, 3, 4, 0, 7, 8, 5] },
    76: { activation: 0.4910, instances: [9, 1, 6, 3, 2, 0, 4, 8, 7, 5] },
    501: { activation: 0.4885, instances: [6, 2, 7, 3, 9, 8, 0, 4, 1, 5] },
    787: { activation: 0.4789, instances: [1, 6, 9, 2, 5, 8, 3, 4, 7, 0] },
    210: { activation: 0.4786, instances: [9, 3, 7, 4, 6, 1, 0, 5, 2, 8] },
    829: { activation: 0.4563, instances: [6, 7, 2, 5, 9, 3, 1, 8, 0, 4] },
    836: { activation: 0.4512, instances: [2, 6, 1, 0, 3, 9, 7, 4, 5, 8] },
    616: { activation: 0.4393, instances: [1, 9, 6, 3, 0, 5, 4, 7, 2, 8] },
    376: { activation: 0.4325, instances: [1, 6, 3, 7, 9, 0, 4, 5, 2, 8] },
    443: { activation: 0.4273, instances: [6, 2, 9, 7, 1, 3, 4, 0, 8, 5] }
};

// Feature viewer functionality
function updateFeatureDisplay() {
    const featureSelect = document.getElementById('feature-select');
    
    if (!featureSelect) return;
    
    const selectedFeature = featureSelect.value;
    const data = featureData[selectedFeature];
    
    if (!data) return;
    
    // Update activation value
    const activationSpan = document.getElementById('current-activation');
    if (activationSpan) {
        activationSpan.textContent = data.activation;
    }
    
    // Update overlay image
    const overlayImg = document.getElementById('current-overlay');
    if (overlayImg) {
        overlayImg.src = `feature_overlays/feature_${selectedFeature}_overlay.png`;
        overlayImg.onclick = () => openImage(`feature_overlays/feature_${selectedFeature}_overlay.png`);
    }
    
    // Update top instances grid (first 5)
    const topInstancesGrid = document.getElementById('top-instances-grid');
    if (topInstancesGrid) {
        topInstancesGrid.innerHTML = '';
        
        data.instances.slice(0, 5).forEach((instanceId, index) => {
            const instanceCard = document.createElement('div');
            instanceCard.className = 'instance-card';
            instanceCard.innerHTML = `
                <img src="feature_${selectedFeature}/instance_${instanceId}.png" 
                     alt="Instance ${instanceId}"
                     onclick="openImage('feature_${selectedFeature}/instance_${instanceId}.png')">
                <p>Instance ${instanceId}<br>Rank: ${index + 1}</p>
            `;
            topInstancesGrid.appendChild(instanceCard);
        });
    }
    
    // Update bottom instances grid (last 5, in reverse order so lowest is first)
    const bottomInstancesGrid = document.getElementById('bottom-instances-grid');
    if (bottomInstancesGrid) {
        bottomInstancesGrid.innerHTML = '';
        
        data.instances.slice(-5).reverse().forEach((instanceId, index) => {
            const instanceCard = document.createElement('div');
            instanceCard.className = 'instance-card';
            instanceCard.innerHTML = `
                <img src="feature_${selectedFeature}/instance_${instanceId}.png" 
                     alt="Instance ${instanceId}"
                     onclick="openImage('feature_${selectedFeature}/instance_${instanceId}.png')">
                <p>Instance ${instanceId}<br>Rank: ${10 - index}</p>
            `;
            bottomInstancesGrid.appendChild(instanceCard);
        });
    }
}

function showGenericFeature(featureId) {
    // This function is no longer needed since we use a single display
    // But keeping it for compatibility
    updateFeatureDisplay();
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Set up tab event listeners
    const tabButtons = document.querySelectorAll('.tab-button');
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            const tabName = this.getAttribute('data-tab');
            if (tabName) {
                switchTab(tabName);
            }
        });
    });
    
    // Set up feature viewer event listeners
    const featureSelect = document.getElementById('feature-select');
    
    if (featureSelect) {
        featureSelect.addEventListener('change', updateFeatureDisplay);
    }
    
    // Initialize display
    updateFeatureDisplay();
});

// Utility function to open images in new tab
function openImage(imagePath) {
    window.open(imagePath, '_blank');
} 