

// ========================================
// API CLIENT - Direct Integration
// ========================================

class APIClient {
    constructor(baseURL = 'http://localhost:8000') {
        this.baseURL = baseURL;
    }

    async fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => {
                const base64 = reader.result.split(',')[1];
                resolve(base64);
            };
            reader.onerror = error => reject(error);
        });
    }

    async processLayer1(file) {
        const csvData = await this.fileToBase64(file);
        const response = await fetch(`${this.baseURL}/api/v1/layer1/process`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ csv_data: csvData, filename: file.name })
        });
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        return await response.json();
    }

    async processLayer2(file, options = {}) {
        const csvData = await this.fileToBase64(file);
        const response = await fetch(`${this.baseURL}/api/v1/layer2/process`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ csv_data: csvData, filename: file.name, ...options })
        });
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        return await response.json();
    }

    async processLayer3(file, config = {}) {
        const csvData = await this.fileToBase64(file);
        const response = await fetch(`${this.baseURL}/api/v1/layer3/process`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ csv_data: csvData, filename: file.name, ...config })
        });
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        return await response.json();
    }

    async searchChunks(processingId, query, options = {}) {
        const response = await fetch(`${this.baseURL}/api/v1/search/${processingId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                model_name: options.model_name || 'all-MiniLM-L6-v2',
                top_k: options.top_k || 5,
                similarity_metric: options.similarity_metric || 'cosine'
            })
        });
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        return await response.json();
    }

    async downloadFile(fileId, filename) {
        const response = await fetch(`${this.baseURL}/api/v1/download/${fileId}`);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = filename || fileId;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        return true;
    }

    async checkHealth() {
        try {
            const response = await fetch(`${this.baseURL}/api/v1/health`);
            return await response.json();
        } catch (error) {
            return { status: 'unhealthy', error: error.message };
        }
    }

    // Step-by-step processing methods
    async processStepPreprocessing(file) {
        const csvData = await this.fileToBase64(file);
        const response = await fetch(`${this.baseURL}/api/v1/step/preprocessing`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ csv_data: csvData, filename: file.name })
        });
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        return await response.json();
    }

    async processStepChunking(file) {
        const csvData = await this.fileToBase64(file);
        const response = await fetch(`${this.baseURL}/api/v1/step/chunking`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ csv_data: csvData, filename: file.name })
        });
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        return await response.json();
    }

    async processStepEmbedding(file) {
        const csvData = await this.fileToBase64(file);
        const response = await fetch(`${this.baseURL}/api/v1/step/embedding`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ csv_data: csvData, filename: file.name })
        });
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        return await response.json();
    }

    async processStepStoring(file) {
        const csvData = await this.fileToBase64(file);
        const response = await fetch(`${this.baseURL}/api/v1/step/storing`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ csv_data: csvData, filename: file.name })
        });
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        return await response.json();
    }
}

// Initialize API Client
const apiClient = new APIClient();

// Debug helper function
function debugLog(message, data = null) {
    console.log(`🔍 [DEBUG] ${message}`, data || '');
}

// Add global error handler for unhandled errors
window.addEventListener('error', function(event) {
    console.error('🚨 Global Error:', event.error);
    console.error('🚨 Error Details:', {
        message: event.message,
        filename: event.filename,
        line: event.lineno,
        column: event.colno
    });
});

// Add unhandled promise rejection handler
window.addEventListener('unhandledrejection', function(event) {
    console.error('🚨 Unhandled Promise Rejection:', event.reason);
    event.preventDefault(); // Prevent the default browser behavior
});

// ========================================
// Global variables
let currentLayer = 1;
let isProcessing = false;
let uploadedFile = null;
let processedData = null;
let processingStartTime = null;
let stepTimers = {};
let stepStartTimes = {};
let layerSelected = true; // Default to true for Fast Mode
let fileUploaded = false;

// Deep Config specific variables
let csvData = null;
let csvHeaders = [];
let currentPreviewData = null;
let deepConfigWorkflowStep = 'upload'; // upload, default-preprocessing, manual-preprocessing, metadata, chunking, embeddings, storing, retrieval
let manualPreprocessingStep = 'datatype'; // datatype, null, duplicate, text
let selectedNumericColumns = [];
let selectedCategoricalColumns = [];
let chunkingComplete = false;
let embeddingsComplete = false;

// File upload handling
function handleFileUpload(event) {
    console.log('handleFileUpload called');
    const file = event.target.files[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.csv')) {
        alert('Please upload a CSV file.');
        return;
    }

    uploadedFile = file;
    const uploadArea = document.getElementById('file-upload-area');
    
    // Update upload area to show success with tick
    uploadArea.classList.add('uploaded');
    uploadArea.innerHTML = `
        <div style="display: flex; align-items: center; justify-content: center; gap: 10px;">
            <div style="width: 20px; height: 20px; background: #10b981; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-size: 12px;">✓</div>
            <h3 style="font-size: 20px; font-weight: 700; margin: 0; color: var(--text-primary);">File Uploaded Successfully!</h3>
        </div>
        <p style="color: var(--text-secondary); margin-top: 8px; font-size: 14px;">${file.name}</p>
        <input type="file" id="csvFile" accept=".csv" style="display: none;" onchange="handleFileUpload(event)">
    `;

    // Update sidebar stats
    document.getElementById('file-size').textContent = `${(file.size / 1024 / 1024).toFixed(1)}MB`;
    
    // Mark file as uploaded
    fileUploaded = true;
    
    // Hide layer selection and file upload after file upload
    hideLayerSelection();
    hideFileUploadSection();
    
    // Show sidebar immediately after file upload
    showSidebarAfterUpload();
    
    // Show processing pipeline immediately after file upload
    showProcessingPipelineAfterUpload();
    
    // Mark upload step as completed
    updateStepStatus('step-upload', 'completed');
    
    // Check if we should show processing pipeline
    checkAndShowProcessingPipeline();
    
    // Show action buttons after file upload (except for Layer 3)
    if (currentLayer !== 3) {
        const actionButtons = document.querySelector('.action-buttons');
        const actionSection = document.querySelector('.action-section');
        
        if (actionButtons) {
            actionButtons.style.display = 'flex';
            actionButtons.style.visibility = 'visible';
            actionButtons.style.opacity = '1';
        }
        if (actionSection) {
            actionSection.style.display = 'flex';
        }
    }
    
    // Handle Deep Config mode
    if (currentLayer === 3) {
        handleDeepConfigFileUpload(file);
    }
}


// Drag and drop functionality
function setupDragDrop() {
    const uploadArea = document.getElementById('file-upload-area');
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            const file = files[0];
            if (file.name.toLowerCase().endsWith('.csv')) {
                document.getElementById('csvFile').files = files;
                handleFileUpload({ target: { files: [file] } });
            } else {
                alert('Please upload a CSV file.');
            }
        }
    });
}

// Hide layer selection after file upload
function hideLayerSelection() {
    const layerSelection = document.querySelector('.layer-selection');
    if (layerSelection) {
        layerSelection.style.display = 'none';
    }
}

// Hide file upload section after file upload
function hideFileUploadSection() {
    const fileUploadSection = document.querySelector('.file-upload-section');
    if (fileUploadSection) {
        fileUploadSection.style.display = 'none';
    }
}

// Show sidebar immediately after file upload
function showSidebarAfterUpload() {
    const sidebar = document.querySelector('.sidebar');
    if (sidebar) {
        sidebar.classList.add('show');
    }
}

// Show processing pipeline immediately after file upload
function showProcessingPipelineAfterUpload() {
    const pipelineSection = document.getElementById('processing-pipeline-section');
    if (pipelineSection) {
        pipelineSection.classList.add('show');
    }
}

// Check if we should show processing pipeline
function checkAndShowProcessingPipeline() {
    if (layerSelected && fileUploaded) {
        // Show sidebar first
        const sidebar = document.querySelector('.sidebar');
        if (sidebar) {
            sidebar.classList.add('show');
        }
        
        // Show processing pipeline based on current layer
        if (currentLayer === 1) {
            // Layer 1 (Fast Mode)
            const pipelineSection = document.getElementById('processing-pipeline-section');
            if (pipelineSection) {
                pipelineSection.classList.add('show');
            }
        } else if (currentLayer === 2) {
            // Layer 2
            const pipelineSectionLayer2 = document.getElementById('processing-pipeline-section-layer2');
            if (pipelineSectionLayer2) {
                pipelineSectionLayer2.classList.add('show');
            }
        }
    }
}

// Layer selection
function selectLayer(layer) {
    console.log('selectLayer called with layer:', layer);
    currentLayer = layer;
    layerSelected = true;
    
    // Update layer cards
    document.querySelectorAll('.layer-card').forEach((card, index) => {
        card.classList.toggle('active', index + 1 === layer);
    });

    // Update content sections
    document.querySelectorAll('.content-section').forEach((section, index) => {
        section.classList.toggle('active', index + 1 === layer);
    });
    
    // Hide action buttons for Deep Config (Layer 3)
    const actionButtons = document.querySelector('.action-buttons');
    const actionSection = document.querySelector('.action-section');
    
    // Add/remove body class for CSS targeting
    document.body.classList.remove('layer-1', 'layer-2', 'layer-3');
    document.body.classList.add(`layer-${layer}`);
    
    if (layer === 3) {
        // Hide action buttons completely for Deep Config
        if (actionButtons) {
            actionButtons.style.display = 'none !important';
            actionButtons.style.visibility = 'hidden';
            actionButtons.style.opacity = '0';
        }
        if (actionSection) {
            actionSection.style.display = 'none !important';
        }
    } else {
        // Show action buttons for other layers only if file is uploaded
        if (fileUploaded) {
            if (actionButtons) {
                actionButtons.style.display = 'flex';
                actionButtons.style.visibility = 'visible';
                actionButtons.style.opacity = '1';
            }
            if (actionSection) {
                actionSection.style.display = 'flex';
            }
        } else {
            // Hide action buttons if no file uploaded yet
            if (actionButtons) {
                actionButtons.style.display = 'none';
                actionButtons.style.visibility = 'hidden';
                actionButtons.style.opacity = '0';
            }
            if (actionSection) {
                actionSection.style.display = 'none';
            }
        }
    }
    
    // Check if we should show processing pipeline
    checkAndShowProcessingPipeline();
}

// Update range slider values
function updateRangeValue(sliderId, valueId) {
    const slider = document.getElementById(sliderId);
    const valueSpan = document.getElementById(valueId);
    
    let value = slider.value;
    
    // Format value based on the slider type
    if (sliderId.includes('overlap')) {
        value += '%';
    } else if (sliderId.includes('threshold') || sliderId.includes('temperature')) {
        value = parseFloat(value).toFixed(2);
    } else if (sliderId.includes('cache-size')) {
        value += 'MB';
    }
    
    valueSpan.textContent = value;
}

// Step status management
function updateStepStatus(stepId, status) {
    const step = document.getElementById(stepId);
    if (!step) return;
    
    const statusTextElement = document.getElementById(`status-text-${stepId.replace('step-', '')}`);
    const timingElement = document.getElementById(`timing-${stepId.replace('step-', '')}`);
    
    // Remove all status classes
    step.classList.remove('active', 'completed', 'error');
    
    switch (status) {
        case 'active':
            step.classList.add('active');
            if (statusTextElement) {
                statusTextElement.textContent = 'Processing';
            }
            if (timingElement) {
                timingElement.textContent = '';
            }
            break;
        case 'completed':
            step.classList.add('completed');
            // Don't clear the completion text - let the timer system handle it
            break;
        case 'error':
            step.classList.add('error');
            if (statusTextElement) {
                statusTextElement.textContent = 'Error';
            }
            if (timingElement) {
                timingElement.textContent = '';
            }
            break;
        default:
            if (statusTextElement) {
                statusTextElement.textContent = '';
            }
            if (timingElement) {
                timingElement.textContent = '';
            }
    }
}

// Timer management functions
function startStepTimer(stepId) {
    const stepName = stepId.replace('step-', '');
    const timingElement = document.getElementById(`timing-${stepName}`);
    
    stepStartTimes[stepId] = Date.now();
    timingElement.textContent = 'Executing...';
    
    // Clear any existing timer
    if (stepTimers[stepId]) {
        clearInterval(stepTimers[stepId]);
    }
    
    // Start live timer
    stepTimers[stepId] = setInterval(() => {
        const elapsed = Math.floor((Date.now() - stepStartTimes[stepId]) / 1000);
        timingElement.textContent = `Executing... ${elapsed}s`;
    }, 1000);
}

function stopStepTimer(stepId, isError = false) {
    const stepName = stepId.replace('step-', '');
    const timingElement = document.getElementById(`timing-${stepName}`);
    
    if (stepTimers[stepId]) {
        clearInterval(stepTimers[stepId]);
        delete stepTimers[stepId];
    }
    
    if (stepStartTimes[stepId]) {
        const elapsed = Math.floor((Date.now() - stepStartTimes[stepId]) / 1000);
        timingElement.textContent = isError ? `Failed in ${elapsed}s` : `Completed in ${elapsed}s`;
        delete stepStartTimes[stepId];
    }
}

// Reset all step timers
function resetAllTimers() {
    Object.keys(stepTimers).forEach(stepId => {
        clearInterval(stepTimers[stepId]);
        delete stepTimers[stepId];
    });
    stepStartTimes = {};
    
    // Reset all timing displays
    const stepNames = ['upload', 'analyze', 'preprocess', 'chunking', 'embedding', 'storage', 'retrieval'];
    stepNames.forEach(stepName => {
        const timingElement = document.getElementById(`timing-${stepName}`);
        if (timingElement) {
            timingElement.textContent = '';
        }
    });
}

// Start processing
async function startProcessing() {
    if (!uploadedFile) {
        alert('Please upload a CSV file first!');
        return;
    }

    if (isProcessing) {
        alert('Processing is already in progress!');
        return;
    }

    isProcessing = true;
    processingStartTime = Date.now();
    
    // Reset all timers
    resetAllTimers();
    
    // Show processing indicator
    const processBtn = document.getElementById('process-btn');
    if (processBtn) {
        processBtn.disabled = true;
        processBtn.textContent = 'Processing...';
    }

    // Hide search section if it exists
    const searchSection = document.getElementById('search-section');
    if (searchSection) {
        searchSection.classList.remove('show');
    }

    try {
        console.log(`🚀 Starting REAL API processing - Layer ${currentLayer}`);
        
        // Check API health
        console.log('🏥 Checking API health...');
        const health = await apiClient.checkHealth();
        console.log('🏥 Health response:', health);
        
        if (health.status !== 'healthy') {
            throw new Error(`FastAPI backend is not healthy: ${JSON.stringify(health)}`);
        }
        console.log('✅ API is healthy, proceeding with processing');

        let result;
        
        // Skip analysis step - start directly from preprocessing
        updateStepStatus('step-preprocess', 'active');
        startStepTimer('step-preprocess', 'Preprocessing');
        
        const progressText = document.getElementById('progress-text');
        if (progressText) {
            progressText.textContent = 'Starting Preprocessing...';
        }
        
        // Process based on current layer with REAL API calls
        if (currentLayer === 1) {
            console.log('📊 Processing with Layer 1 (Fast Mode) - DYNAMIC STEP-BY-STEP');
            console.log('🔧 Real-time processing with individual API calls per step');
            
            result = await processDynamicStepByStep(uploadedFile);
            
        } else if (currentLayer === 2) {
            console.log('⚙️ Processing with Layer 2 (Config Mode)');
            
            // Get Layer 2 configuration from UI
            const config = {
                chunking_method: document.getElementById('chunking-method')?.value || 'semantic',
                embedding_model: document.getElementById('embedding-model')?.value || 'all-MiniLM-L6-v2',
                batch_size: parseInt(document.getElementById('batch-size')?.value) || 64
            };
            
            result = await apiClient.processLayer2(uploadedFile, config);
            
        } else if (currentLayer === 3) {
            console.log('🔧 Processing with Layer 3 (Deep Config Mode)');
            
            // For Layer 3, use the existing deep config workflow
            handleDeepConfigFileUpload(uploadedFile);
            return;
        }
        
        // Handle successful processing
        console.log('🔍 Raw API Response:', result);
        console.log('🔍 Response type:', typeof result);
        console.log('🔍 Response keys:', result ? Object.keys(result) : 'null');
        
        if (result && result.success) {
            console.log('✅ Processing successful:', result);
            
            // For dynamic processing, we already handled everything in processDynamicStepByStep
            // So we don't need to call handleRealProcessingSuccess
            if (currentLayer === 1) {
                console.log('✅ Dynamic processing completed - downloads and search already enabled');
                
                // Update process button
                const processBtn = document.getElementById('process-btn');
                if (processBtn) {
                    processBtn.disabled = false;
                    processBtn.textContent = 'Start Processing';
                }
                
                // Update progress bar to 100%
                const progressBar = document.getElementById('progress-bar');
                if (progressBar) {
                    progressBar.style.width = '100%';
                }
                
                return; // Exit early - don't call the old static system
            }
            
            // For Layer 2 and 3, use the old system
            await handleRealProcessingSuccess(result);
        } else if (result && result.detail) {
            // FastAPI error response format
            console.error('❌ FastAPI Error:', result.detail);
            throw new Error(`API Error: ${result.detail}`);
        } else {
            const errorMsg = result?.error || result?.message || 'API processing failed - no success field in response';
            console.error('❌ Processing failed:', result);
            throw new Error(errorMsg);
        }
        
    } catch (error) {
        console.error('❌ API Processing failed:', error);
        alert(`Processing failed: ${error.message}\n\nPlease check if FastAPI backend is running on http://localhost:8000`);
        
        // Reset processing state
        isProcessing = false;
        const processBtn = document.getElementById('process-btn');
        if (processBtn) {
            processBtn.disabled = false;
            processBtn.textContent = 'Start Processing';
        }
        updateStepStatus('step-preprocess', 'error');
    }
}

// Handle successful real API processing
async function handleRealProcessingSuccess(result) {
    try {
        console.log('🔍 handleRealProcessingSuccess called with:', result);
        
        const progressBar = document.getElementById('progress-bar');
        const progressText = document.getElementById('progress-text');
        
        // Store result globally
        processedData = result;
    
    // Process each step with real timing based on backend response
    const steps = ['step-preprocess', 'step-chunking', 'step-embedding', 'step-storage', 'step-retrieval'];
    const stepNames = [
        'Preprocessing',
        'Semantic Chunking', 
        'Generating Embeddings',
        'Storing in Database',
        'Retrieval System Ready'
    ];
    
    // Get timing data from backend response
    const summary = result.processing_summary || {};
    const totalTime = summary.processing_time_seconds || 0;
    
    // Simulate step-by-step progression with realistic timing
    for (let i = 0; i < steps.length; i++) {
        // Start step timer
        startStepTimer(steps[i], stepNames[i]);
        updateStepStatus(steps[i], 'active');
        
        if (progressText) {
            progressText.textContent = `Processing: ${stepNames[i]}`;
        }
        
        // Calculate realistic step duration based on total time
        let stepDuration;
        if (i === 0) stepDuration = Math.max(2, Math.floor(totalTime * 0.2)); // Preprocessing: 20%
        else if (i === 1) stepDuration = Math.max(3, Math.floor(totalTime * 0.4)); // Chunking: 40%
        else if (i === 2) stepDuration = Math.max(2, Math.floor(totalTime * 0.3)); // Embedding: 30%
        else if (i === 3) stepDuration = Math.max(1, Math.floor(totalTime * 0.08)); // Storing: 8%
        else stepDuration = Math.max(1, Math.floor(totalTime * 0.02)); // Retrieval: 2%
        
        // Wait for step duration
        await new Promise(resolve => setTimeout(resolve, stepDuration * 1000));
        
        // Stop step timer and mark as completed
        stopStepTimer(steps[i], stepNames[i]);
        updateStepStatus(steps[i], 'completed');
        
        const progress = ((i + 1) / steps.length) * 100;
        if (progressBar) {
            progressBar.style.width = progress + '%';
        }
        
        // Delay between steps for visual effect
        await new Promise(resolve => setTimeout(resolve, 600));
    }
    
    // Update processing time and stats with REAL data
    const processingTime = result.processing_summary?.processing_time_seconds || 
                          ((Date.now() - processingStartTime) / 1000);
    
    try {
        updateProcessingTime(processingTime);
        console.log('✅ Processing time updated');
    } catch (error) {
        console.error('❌ Error updating processing time:', error);
    }
    
    try {
        updateSidebarWithRealStats(result);
        console.log('✅ Sidebar stats updated');
    } catch (error) {
        console.error('❌ Error updating sidebar stats:', error);
    }
    
    // Show download buttons for REAL files
    try {
        console.log('🔍 Download links:', result.download_links);
        showRealDownloadButtons(result.download_links);
        console.log('✅ Download buttons created');
    } catch (error) {
        console.error('❌ Error creating download buttons:', error);
    }
    
    // Enable expandable search interface
    try {
        console.log('🔍 Processing ID:', result.processing_id);
        enableExpandableSearchInterface(result.processing_id);
        console.log('✅ Expandable search interface enabled');
    } catch (error) {
        console.error('❌ Error enabling expandable search interface:', error);
    }
    
    // Complete processing
    if (progressText) {
        progressText.textContent = 'Processing Complete!';
    }
    
    const processBtn = document.getElementById('process-btn');
    if (processBtn) {
        processBtn.textContent = 'Processed';
        processBtn.disabled = true;
    }
    
    isProcessing = false;
    
    // Ensure search interface is visible
    setTimeout(() => {
        const realSearchSection = document.getElementById('search-section');
        if (realSearchSection) {
            realSearchSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
            console.log('✅ Real search interface scrolled into view');
        } else {
            console.log('⚠️ No real search interface found, showing demo');
            showQuerySection();
        }
    }, 500);
    
    console.log('🎉 Real processing completed successfully!');
    
    } catch (error) {
        console.error('❌ Error in handleRealProcessingSuccess:', error);
        alert(`Post-processing error: ${error.message}`);
    }
}

// Update sidebar with real API data
function updateSidebarWithRealStats(result) {
    const summary = result.processing_summary;
    if (!summary) return;
    
    // Update stats with real data
    const stats = {
        rows: summary.input_data?.total_rows || 0,
        columns: summary.input_data?.total_columns || 0,
        chunks: summary.chunking_results?.total_chunks || 0,
        embeddings: summary.embedding_results?.total_embeddings || 0,
        time: `${summary.processing_time_seconds || 0}s`
    };
    
    Object.entries(stats).forEach(([key, value]) => {
        const element = document.querySelector(`[data-stat="${key}"]`);
        if (element) {
            element.textContent = value;
        }
    });
    
    console.log('📊 Updated sidebar with real stats:', stats);
}

// Dynamic Step-by-Step Processing Function
async function processDynamicStepByStep(file) {
    console.log('🚀 Starting DYNAMIC step-by-step processing');
    
    const progressText = document.getElementById('progress-text');
    let totalProcessingTime = 0;
    let allResults = {};
    
    try {
        // Step 1: Preprocessing
        console.log('📊 Step 1: Dynamic Preprocessing');
        startStepTimer('step-preprocess', 'Preprocessing');
        updateStepStatus('step-preprocess', 'active');
        if (progressText) progressText.textContent = 'Processing: Real-time Preprocessing...';
        
        const preprocessResult = await apiClient.processStepPreprocessing(file);
        const preprocessTime = stopStepTimer('step-preprocess', 'Preprocessing');
        updateStepStatus('step-preprocess', 'completed');
        totalProcessingTime += preprocessResult.processing_time_seconds;
        allResults.preprocessing = preprocessResult;
        
        console.log(`✅ Preprocessing completed in ${preprocessResult.processing_time_seconds}s`);
        
        // Step 2: Chunking
        console.log('📊 Step 2: Dynamic Chunking');
        startStepTimer('step-chunking', 'Chunking');
        updateStepStatus('step-chunking', 'active');
        if (progressText) progressText.textContent = 'Processing: Real-time Semantic Chunking...';
        
        const chunkingResult = await apiClient.processStepChunking(file);
        const chunkingTime = stopStepTimer('step-chunking', 'Chunking');
        updateStepStatus('step-chunking', 'completed');
        totalProcessingTime += chunkingResult.processing_time_seconds;
        allResults.chunking = chunkingResult;
        
        console.log(`✅ Chunking completed in ${chunkingResult.processing_time_seconds}s - ${chunkingResult.total_chunks} chunks`);
        
        // Step 3: Embedding
        console.log('📊 Step 3: Dynamic Embedding');
        startStepTimer('step-embedding', 'Embedding');
        updateStepStatus('step-embedding', 'active');
        if (progressText) progressText.textContent = 'Processing: Real-time Embedding Generation...';
        
        const embeddingResult = await apiClient.processStepEmbedding(file);
        const embeddingTime = stopStepTimer('step-embedding', 'Embedding');
        updateStepStatus('step-embedding', 'completed');
        totalProcessingTime += embeddingResult.processing_time_seconds;
        allResults.embedding = embeddingResult;
        
        console.log(`✅ Embedding completed in ${embeddingResult.processing_time_seconds}s - ${embeddingResult.total_embeddings} embeddings`);
        
        // Step 4: Real Backend Processing (Storing + Downloads)
        console.log('📊 Step 4: Real Backend Processing - Storing & Downloads');
        
        // Start storing timer for real backend processing
        startStepTimer('step-storage', 'Storing');
        updateStepStatus('step-storage', 'active');
        if (progressText) progressText.textContent = 'Processing: Real-time Vector Storage...';
        
        // Call the REAL Layer 1 API which does actual storing and creates download files
        console.log('📁 Calling REAL Layer 1 API for storing and downloads...');
        const realFullResult = await apiClient.processLayer1(file);
        
        // Stop storing timer with real processing time
        const storingTime = stopStepTimer('step-storage', 'Storing');
        updateStepStatus('step-storage', 'completed');
        
        // Add the real backend processing time
        const backendProcessingTime = (realFullResult?.processing_summary?.processing_time_seconds || storingTime);
        totalProcessingTime += backendProcessingTime;
        
        console.log(`✅ REAL storing completed in ${backendProcessingTime}s`);
        console.log(`📊 Storing: ${storingTime}s`);
        
        // Mark retrieval as ready (but no timing yet - timing starts when user searches)
        updateStepStatus('step-retrieval', 'completed');
        const retrievalStatusText = document.getElementById('status-text-retrieval');
        if (retrievalStatusText) {
            retrievalStatusText.textContent = 'Ready';
        }
        
        // Store the full result for later use
        allResults.fullBackendResult = realFullResult;
        allResults.storing = {
            storage_type: realFullResult?.processing_summary?.storage_type || 'chroma',
            similarity_metric: realFullResult?.processing_summary?.similarity_metric || 'cosine',
            processing_time_seconds: storingTime
        };
        // Retrieval timing will be added when user actually searches
        
        // Create a consolidated result that matches the expected format
        const consolidatedResult = {
            success: true,
            processing_id: `dynamic_${Date.now()}`,
            timestamp: new Date().toISOString(),
            processing_summary: {
                processing_id: `dynamic_${Date.now()}`,
                layer_mode: "fast",
                timestamp: new Date().toISOString(),
                processing_time_seconds: totalProcessingTime,
                input_data: {
                    total_rows: preprocessResult.input_rows,
                    total_columns: preprocessResult.input_columns,
                    filename: file.name
                },
                chunking_results: {
                    method: chunkingResult.method,
                    total_chunks: chunkingResult.total_chunks,
                    quality_score: chunkingResult.quality_score,
                    quality_rating: chunkingResult.quality_rating
                },
                embedding_results: {
                    model_used: embeddingResult.model_used,
                    vector_dimension: embeddingResult.vector_dimension,
                    total_embeddings: embeddingResult.total_embeddings
                },
                storage_results: {
                    storage_type: realFullResult?.processing_summary?.storage_type || 'chroma',
                    similarity_metric: realFullResult?.processing_summary?.similarity_metric || 'cosine'
                },
                performance_metrics: {
                    rows_per_second: preprocessResult.input_rows / totalProcessingTime,
                    chunks_per_second: chunkingResult.total_chunks / totalProcessingTime
                }
            },
            download_links: {
                // Note: For full functionality, you'd need to call the original Layer1 API
                // after step-by-step processing to get actual download links
                chunks_csv: { message: "Dynamic processing - use full Layer1 API for downloads" },
                embeddings_json: { message: "Dynamic processing - use full Layer1 API for downloads" }
            },
            search_endpoint: `/api/v1/search/dynamic_${Date.now()}`,
            message: `Dynamic step-by-step processing completed in ${totalProcessingTime.toFixed(2)}s`,
            dynamic_results: allResults
        };
        
        console.log('🎉 DYNAMIC step-by-step processing completed!');
        console.log('📊 Total time:', totalProcessingTime.toFixed(2), 's');
        console.log('📊 Results:', consolidatedResult);
        
        // Update final processing status
        if (progressText) {
            progressText.textContent = `Dynamic processing completed in ${totalProcessingTime.toFixed(2)}s`;
        }
        
        // Update total processing time in sidebar
        updateProcessingTime(totalProcessingTime);
        
        // Use the full result we already got from real backend processing
        console.log('📁 Using real backend result for download links...');
        
        const backendResult = allResults.fullBackendResult;
        
        if (backendResult && backendResult.success) {
            console.log('✅ Using full API result with download links:', backendResult);
            
            // Show real download buttons
            if (backendResult.download_links) {
                showRealDownloadButtons(backendResult.download_links);
            }
            
            // Enable expandable search interface
            if (backendResult.processing_id) {
                enableExpandableSearchInterface(backendResult.processing_id);
            }
            
            // Update the consolidated result with real data
            consolidatedResult.download_links = backendResult.download_links;
            consolidatedResult.search_endpoint = backendResult.search_endpoint;
            consolidatedResult.processing_id = backendResult.processing_id;
            
        } else {
            console.warn('⚠️ No full API result available');
            if (progressText) {
                progressText.textContent = 'Processing completed with timing - downloads may not be available';
            }
        }
        
        return consolidatedResult;
        
    } catch (error) {
        console.error('❌ Dynamic step-by-step processing failed:', error);
        
        // Stop any running timers
        ['step-preprocess', 'step-chunking', 'step-embedding', 'step-storage', 'step-retrieval'].forEach(stepId => {
            if (stepTimers[stepId]) {
                clearInterval(stepTimers[stepId]);
                delete stepTimers[stepId];
            }
            updateStepStatus(stepId, 'error');
        });
        
        throw error;
    }
}

// Step timing functions
function startStepTimer(stepId, stepName) {
    const now = Date.now();
    stepStartTimes[stepId] = now;
    stepTimers[stepId] = setInterval(() => {
        const elapsed = Math.floor((Date.now() - now) / 1000);
        updateStepLiveTime(stepId, elapsed);
    }, 1000);
    
    console.log(`⏱️ Started timer for ${stepName} (${stepId})`);
}

function stopStepTimer(stepId, stepName) {
    if (stepTimers[stepId]) {
        clearInterval(stepTimers[stepId]);
        delete stepTimers[stepId];
    }
    
    if (stepStartTimes[stepId]) {
        const elapsed = Math.floor((Date.now() - stepStartTimes[stepId]) / 1000);
        updateStepCompletionTime(stepId, elapsed, stepName);
        delete stepStartTimes[stepId];
        console.log(`⏱️ ${stepName} completed in ${elapsed}s`);
        return elapsed;
    }
    return 0;
}

function updateStepLiveTime(stepId, seconds) {
    const stepElement = document.getElementById(stepId);
    if (stepElement) {
        // Update status text
        const statusTextElement = document.getElementById(`status-text-${stepId.replace('step-', '')}`);
        if (statusTextElement && stepElement.classList.contains('active')) {
            statusTextElement.textContent = 'Processing';
        }
        
        // Update timing
        const timingElement = document.getElementById(`timing-${stepId.replace('step-', '')}`);
        if (timingElement && stepElement.classList.contains('active')) {
            timingElement.textContent = `${seconds}s`;
        }
    }
}

function updateStepCompletionTime(stepId, seconds, stepName) {
    const stepElement = document.getElementById(stepId);
    if (stepElement) {
        // Update status text
        const statusTextElement = document.getElementById(`status-text-${stepId.replace('step-', '')}`);
        if (statusTextElement) {
            statusTextElement.textContent = 'Completed';
        }
        
        // Update timing display
        const timingElement = document.getElementById(`timing-${stepId.replace('step-', '')}`);
        if (timingElement) {
            timingElement.textContent = `${seconds}s`;
        }
    }
}

// Update processing time display
function updateProcessingTime(processingTimeSeconds) {
    const timeElement = document.querySelector('[data-stat="time"]');
    if (timeElement) {
        // Format time nicely
        const seconds = Math.round(processingTimeSeconds * 100) / 100; // Round to 2 decimal places
        timeElement.textContent = `${seconds}s`;
    }
    
    // Also update any other time displays
    const processTimeElement = document.getElementById('process-time');
    if (processTimeElement) {
        const seconds = Math.round(processingTimeSeconds * 100) / 100;
        processTimeElement.textContent = `${seconds}s`;
    }
    
    console.log(`⏱️ Processing time updated: ${processingTimeSeconds}s`);
}

// Show real download buttons
function showRealDownloadButtons(downloadLinks) {
    if (!downloadLinks) return;
    
    // Find or create download section
    let downloadSection = document.getElementById('download-section');
    if (!downloadSection) {
        downloadSection = document.createElement('div');
        downloadSection.id = 'download-section';
        downloadSection.className = 'config-card';
        downloadSection.style.cssText = `
            margin: -150px auto 15px auto;
            max-width: 800px;
            width: 100%;
            background: #1d2224;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 12px;
            position: relative;
            z-index: 999;
            height: auto;
            min-height: 100px;
            overflow: visible;
            top: -60px;
        `;
        downloadSection.innerHTML = `
            <div class="config-card-header" style="margin-bottom: 10px;">
                <div class="config-icon">📁</div>
                <div class="config-title">Download Processed Files</div>
            </div>
            <div id="download-buttons" class="form-group" style="display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; min-height: 60px; height: auto; overflow: visible; z-index: 1000;"></div>
        `;
        
        // Add to the active content section - INSERT HIGHER UP, not at the end
        const activeSection = document.querySelector('.content-section.active .content-wrapper') || 
                             document.querySelector('.content-section.active') ||
                             document.querySelector('.main-content');
        
        if (activeSection) {
            // Find a good insertion point - after processing info but before query section
            const existingQuerySection = activeSection.querySelector('#real-query-section');
            const processingComplete = activeSection.querySelector('.processing-complete');
            
            if (existingQuerySection) {
                // Insert before query section
                activeSection.insertBefore(downloadSection, existingQuerySection);
            } else if (processingComplete) {
                // Insert after processing complete message
                processingComplete.parentNode.insertBefore(downloadSection, processingComplete.nextSibling);
            } else {
                // Insert at a higher position - not at the very end
                const children = activeSection.children;
                if (children.length > 2) {
                    activeSection.insertBefore(downloadSection, children[children.length - 1]);
                } else {
                    activeSection.appendChild(downloadSection);
                }
            }
            console.log('✅ Download section added to:', activeSection.className);
        } else {
            console.error('❌ No active section found for download buttons');
            // Fallback: add to body
            document.body.appendChild(downloadSection);
        }
    }
    
    // Create download buttons
    const buttonsContainer = document.getElementById('download-buttons');
    if (!buttonsContainer) {
        console.error('❌ download-buttons container not found');
        return;
    }
    buttonsContainer.innerHTML = '';
    
    const buttonLabels = {
        chunks_csv: '📊 Chunks CSV',
        embeddings_json: '🧠 Embeddings JSON', 
        metadata_json: '📋 Metadata JSON',
        summary_json: '📈 Summary JSON',
        results_zip: '📦 All Files (ZIP)'
    };
    
    Object.entries(downloadLinks).forEach(([type, linkInfo]) => {
        try {
            const button = document.createElement('button');
            button.className = 'btn btn-secondary';
            button.innerHTML = buttonLabels[type] || `📄 ${type}`;
            button.style.minWidth = '150px';
            button.onclick = (event) => downloadRealFile(linkInfo.file_id, linkInfo.url, event);
            buttonsContainer.appendChild(button);
        } catch (error) {
            console.error('❌ Error creating download button:', error);
        }
    });
    
    console.log('📁 Added real download buttons:', Object.keys(downloadLinks));
}

// Download real file
async function downloadRealFile(fileId, url, event) {
    try {
        console.log(`📥 Downloading real file: ${fileId}`);
        const filename = url.split('/').pop();
        await apiClient.downloadFile(fileId, filename);
        
        // Show success message
        const button = event.target;
        const originalText = button.innerHTML;
        button.innerHTML = '✅ Downloaded!';
        button.disabled = true;
        
        setTimeout(() => {
            button.innerHTML = originalText;
            button.disabled = false;
        }, 2000);
        
        console.log(`✅ Successfully downloaded: ${filename}`);
        
    } catch (error) {
        console.error('❌ Download failed:', error);
        alert(`Download failed: ${error.message}`);
    }
}

// Enable real query interface
function enableRealQueryInterface(processingId) {
    // Find or create search section
    let searchSection = document.getElementById('search-section');
    if (!searchSection) {
        searchSection = document.createElement('div');
        searchSection.id = 'search-section';
        searchSection.className = 'config-card';
        searchSection.style.cssText = `
            margin: 20px auto;
            max-width: 800px;
            width: 100%;
            background: #1d2224;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 20px;
            display: block;
        `;
        searchSection.innerHTML = `
            <div class="config-card-header" style="margin-bottom: 15px;">
                <div class="config-icon">🔍</div>
                <div class="config-title">Search Processed Data</div>
            </div>
            <div class="form-group">
                <input type="text" id="real-search-query" class="form-control" 
                       placeholder="Enter your search query..." style="margin-bottom: 10px; width: 100%; padding: 10px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.2); background: rgba(255,255,255,0.05); color: white;">
                <div style="display: flex; gap: 10px; align-items: center; margin-bottom: 10px;">
                    <select id="similarity-metric" class="form-control" style="flex: 1; padding: 8px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.2); background: rgba(255,255,255,0.05); color: white;">
                        <option value="cosine">Cosine Similarity</option>
                        <option value="dot">Dot Product</option>
                        <option value="euclidean">Euclidean Distance</option>
                    </select>
                    <input type="number" id="top-k" class="form-control" value="5" min="1" max="20" 
                           placeholder="Results" style="width: 100px; padding: 8px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.2); background: rgba(255,255,255,0.05); color: white;">
                </div>
                <button onclick="performRealSearch('${processingId}')" class="btn btn-primary" style="width: 100%; padding: 12px; background: #10b981; color: white; border: none; border-radius: 6px; cursor: pointer;">
                    🔍 Search Chunks
                </button>
            </div>
            <div id="real-search-results" class="search-results"></div>
        `;
        
        // Add to active content section
        const activeSection = document.querySelector('.content-section.active .content-wrapper') || 
                             document.querySelector('.content-section.active') ||
                             document.querySelector('.main-content');
        
        if (activeSection) {
            activeSection.appendChild(searchSection);
            console.log('✅ Search section added to:', activeSection.className);
        } else {
            console.error('❌ No active section found for search interface');
            // Fallback: add to body
            document.body.appendChild(searchSection);
        }
    }
    
    // Add Enter key support
    const searchInput = document.getElementById('real-search-query');
    if (searchInput) {
        searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                performRealSearch(processingId);
            }
        });
    }
    
    console.log('🔍 Enabled real query interface for:', processingId);
}

// Enable expandable search interface
function enableExpandableSearchInterface(processingId) {
    // Find or create expandable search section
    let expandableSearchSection = document.getElementById('expandable-search-section');
    if (!expandableSearchSection) {
        expandableSearchSection = document.createElement('div');
        expandableSearchSection.id = 'expandable-search-section';
        expandableSearchSection.style.cssText = `
            margin: -20px auto 15px auto;
            max-width: 800px;
            width: 100%;
            background: #1d2224;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            overflow: hidden;
            position: relative;
            z-index: 998;
            top: -40px;
        `;
        
        expandableSearchSection.innerHTML = `
            <!-- Collapsed Header (Always Visible) -->
            <div id="search-header" class="expandable-header" onclick="toggleSearchSection()" style="
                height: 40px;
                padding: 0 15px;
                display: flex;
                align-items: center;
                cursor: pointer;
                background: #1d2224;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                transition: background 0.3s ease;
            ">
                <div class="config-icon" style="margin-right: 10px;">🔍</div>
                <div class="config-title" style="flex: 1; font-size: 14px; font-weight: 600;">Search Retrieved Chunks</div>
                <div id="search-toggle-icon" style="font-size: 12px; color: #ccc;">▼</div>
            </div>
            
            <!-- Expandable Content (Hidden by default) -->
            <div id="search-content" style="
                display: none;
                padding: 15px;
                background: #1d2224;
            ">
                <!-- Query Input -->
                <div style="margin-bottom: 10px;">
                    <input type="text" id="expandable-search-query" 
                           placeholder="Enter your search query..." 
                           style="
                               width: 100%;
                               height: 50px;
                               padding: 15px;
                               border-radius: 8px;
                               border: 1px solid rgba(255,255,255,0.2);
                               background: rgba(255,255,255,0.05);
                               color: white;
                               font-size: 14px;
                               box-sizing: border-box;
                           ">
                </div>
                
                <!-- Search Controls -->
                <div style="display: flex; gap: 10px; margin-bottom: 10px;">
                    <select id="expandable-similarity-metric" style="
                        flex: 1;
                        height: 40px;
                        padding: 8px;
                        border-radius: 6px;
                        border: 1px solid rgba(255,255,255,0.2);
                        background: rgba(255,255,255,0.05);
                        color: white;
                    ">
                        <option value="cosine">Cosine Similarity</option>
                        <option value="dot">Dot Product</option>
                        <option value="euclidean">Euclidean Distance</option>
                    </select>
                    <input type="number" id="expandable-top-k" value="5" min="1" max="20" 
                           placeholder="Results" style="
                               width: 100px;
                               height: 40px;
                               padding: 8px;
                               border-radius: 6px;
                               border: 1px solid rgba(255,255,255,0.2);
                               background: rgba(255,255,255,0.05);
                               color: white;
                           ">
                    <button onclick="performExpandableSearch('${processingId}')" style="
                        height: 40px;
                        padding: 0 20px;
                        background: #10b981;
                        color: white;
                        border: none;
                        border-radius: 6px;
                        cursor: pointer;
                        font-weight: 600;
                    ">🔍 Search</button>
                </div>
                
                <!-- Results Area -->
                <div id="expandable-search-results" style="
                    height: 200px;
                    border: 1px solid rgba(255,255,255,0.1);
                    border-radius: 8px;
                    background: rgba(255,255,255,0.02);
                    overflow-y: auto;
                    padding: 10px;
                    font-size: 13px;
                "></div>
            </div>
        `;
        
        // Insert after download section
        const downloadSection = document.getElementById('download-section');
        if (downloadSection && downloadSection.parentNode) {
            downloadSection.parentNode.insertBefore(expandableSearchSection, downloadSection.nextSibling);
            console.log('✅ Expandable search section added after download section');
        } else {
            // Fallback: add to active content section
            const activeSection = document.querySelector('.content-section.active .content-wrapper') || 
                                 document.querySelector('.content-section.active') ||
                                 document.querySelector('.main-content');
            if (activeSection) {
                activeSection.appendChild(expandableSearchSection);
                console.log('✅ Expandable search section added to active section');
            }
        }
        
        // Add Enter key listener
        const searchInput = document.getElementById('expandable-search-query');
        if (searchInput) {
            searchInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    performExpandableSearch(processingId);
                }
            });
        }
    }
    
    console.log('🔍 Enabled expandable search interface for:', processingId);
}

// Toggle search section expand/collapse
function toggleSearchSection() {
    const content = document.getElementById('search-content');
    const icon = document.getElementById('search-toggle-icon');
    const header = document.getElementById('search-header');
    
    if (content && icon && header) {
        const isExpanded = content.style.display !== 'none';
        
        if (isExpanded) {
            // Collapse
            content.style.display = 'none';
            icon.textContent = '▼';
            header.style.background = '#1d2224';
        } else {
            // Expand
            content.style.display = 'block';
            icon.textContent = '▲';
            header.style.background = '#252729';
        }
    }
}

// Perform expandable search
async function performExpandableSearch(processingId) {
    const query = document.getElementById('expandable-search-query').value.trim();
    if (!query) {
        alert('Please enter a search query');
        return;
    }
    
    const similarityMetric = document.getElementById('expandable-similarity-metric').value;
    const topK = parseInt(document.getElementById('expandable-top-k').value) || 5;
    
    const resultsContainer = document.getElementById('expandable-search-results');
    if (!resultsContainer) {
        console.error('❌ expandable-search-results container not found');
        return;
    }
    
    // Start retrieval timing in the sidebar
    startStepTimer('step-retrieval', 'Retrieval');
    updateStepStatus('step-retrieval', 'active');
    const retrievalStatusText = document.getElementById('status-text-retrieval');
    if (retrievalStatusText) {
        retrievalStatusText.textContent = 'Searching';
    }
    
    // Start search timing for results display
    const searchStartTime = Date.now();
    let searchTimer = setInterval(() => {
        const elapsed = Math.floor((Date.now() - searchStartTime) / 1000);
        resultsContainer.innerHTML = `<p style="text-align: center; color: #ccc; margin: 80px 0;">🔍 Searching... ${elapsed}s</p>`;
    }, 1000);
    
    resultsContainer.innerHTML = '<p style="text-align: center; color: #ccc; margin: 80px 0;">🔍 Searching... 0s</p>';
    
    try {
        console.log(`🔍 Expandable search: "${query}" with ${similarityMetric} similarity`);
        
        const searchResults = await apiClient.searchChunks(processingId, query, {
            similarity_metric: similarityMetric,
            top_k: topK
        });
        
        // Stop search timer
        clearInterval(searchTimer);
        const searchTime = Math.floor((Date.now() - searchStartTime) / 1000);
        
        // Stop retrieval timing in sidebar
        const retrievalTime = stopStepTimer('step-retrieval', 'Retrieval');
        updateStepStatus('step-retrieval', 'completed');
        
        if (searchResults.success && searchResults.results.length > 0) {
            displayExpandableSearchResults(searchResults.results, searchResults.query, searchTime);
        } else {
            resultsContainer.innerHTML = `<p style="text-align: center; color: #ccc; margin: 80px 0;">No results found for your query. (Completed in ${searchTime}s)</p>`;
        }
        
    } catch (error) {
        clearInterval(searchTimer);
        const searchTime = Math.floor((Date.now() - searchStartTime) / 1000);
        
        // Stop retrieval timing in sidebar (error case)
        const retrievalTime = stopStepTimer('step-retrieval', 'Retrieval');
        updateStepStatus('step-retrieval', 'error');
        
        console.error('❌ Expandable search failed:', error);
        resultsContainer.innerHTML = `<p style="color: red; text-align: center; margin: 80px 0;">Search failed: ${error.message} (Failed after ${searchTime}s)</p>`;
    }
}

// Display expandable search results
function displayExpandableSearchResults(results, query, searchTime = 0) {
    const resultsContainer = document.getElementById('expandable-search-results');
    if (!resultsContainer) {
        console.error('❌ expandable-search-results container not found');
        return;
    }
    
    const resultsHTML = `
        <div style="margin-bottom: 15px; padding: 10px; background: rgba(16, 185, 129, 0.1); border-radius: 6px; border-left: 3px solid #10b981;">
            <strong>Search Results for:</strong> "${query}" 
            <span style="color: #10b981;">(${results.length} results in ${searchTime}s)</span>
        </div>
        ${results.map((result, index) => `
            <div style="margin-bottom: 12px; padding: 12px; background: rgba(255,255,255,0.03); border-radius: 6px; border-left: 2px solid #10b981;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <strong style="color: #10b981; font-size: 12px;">Chunk ${index + 1}</strong>
                    <span style="background: rgba(16, 185, 129, 0.2); color: #10b981; padding: 2px 8px; border-radius: 12px; font-size: 11px;">
                        Score: ${(result.similarity_score || result.score || 0).toFixed(3)}
                    </span>
                </div>
                <div style="color: #e5e5e5; font-size: 13px; line-height: 1.4;">
                    ${result.content || result.text || 'No content available'}
                </div>
                ${result.metadata ? `
                    <div style="margin-top: 8px; font-size: 11px; color: #999;">
                        <strong>Metadata:</strong> ${JSON.stringify(result.metadata)}
                    </div>
                ` : ''}
            </div>
        `).join('')}
    `;
    
    resultsContainer.innerHTML = resultsHTML;
    console.log('📋 Displayed expandable search results:', results.length);
}

// Perform real search
async function performRealSearch(processingId) {
    const query = document.getElementById('real-search-query').value.trim();
    if (!query) {
        alert('Please enter a search query');
        return;
    }
    
    const similarityMetric = document.getElementById('similarity-metric').value;
    const topK = parseInt(document.getElementById('top-k').value) || 5;
    
    const resultsContainer = document.getElementById('real-search-results');
    if (!resultsContainer) {
        console.error('❌ real-search-results container not found for search');
        return;
    }
    
    // Start retrieval timing in the sidebar
    startStepTimer('step-retrieval', 'Retrieval');
    updateStepStatus('step-retrieval', 'active');
    const retrievalStatusText = document.getElementById('status-text-retrieval');
    if (retrievalStatusText) {
        retrievalStatusText.textContent = 'Searching';
    }
    
    // Start search timing for results display
    const searchStartTime = Date.now();
    let searchTimer = setInterval(() => {
        const elapsed = Math.floor((Date.now() - searchStartTime) / 1000);
        resultsContainer.innerHTML = `<p>🔍 Searching... ${elapsed}s</p>`;
    }, 1000);
    
    resultsContainer.innerHTML = '<p>🔍 Searching... 0s</p>';
    
    try {
        console.log(`🔍 Real search: "${query}" with ${similarityMetric} similarity`);
        
        const searchResults = await apiClient.searchChunks(processingId, query, {
            similarity_metric: similarityMetric,
            top_k: topK
        });
        
        // Stop search timer
        clearInterval(searchTimer);
        const searchTime = Math.floor((Date.now() - searchStartTime) / 1000);
        
        // Stop retrieval timing in sidebar
        const retrievalTime = stopStepTimer('step-retrieval', 'Retrieval');
        updateStepStatus('step-retrieval', 'completed');
        
        if (searchResults.success && searchResults.results.length > 0) {
            displayRealSearchResults(searchResults.results, searchResults.query, searchTime);
        } else {
            if (resultsContainer) {
                resultsContainer.innerHTML = `<p>No results found for your query. (Completed in ${searchTime}s)</p>`;
            }
        }
        
    } catch (error) {
        clearInterval(searchTimer);
        const searchTime = Math.floor((Date.now() - searchStartTime) / 1000);
        
        // Stop retrieval timing in sidebar (error case)
        const retrievalTime = stopStepTimer('step-retrieval', 'Retrieval');
        updateStepStatus('step-retrieval', 'error');
        
        console.error('❌ Search failed:', error);
        if (resultsContainer) {
            resultsContainer.innerHTML = `<p style="color: red;">Search failed: ${error.message} (Failed after ${searchTime}s)</p>`;
        }
    }
}

// Display real search results
function displayRealSearchResults(results, query, searchTime = 0) {
    const resultsContainer = document.getElementById('real-search-results');
    if (!resultsContainer) {
        console.error('❌ real-search-results container not found');
        return;
    }
    
    const resultsHTML = `
        <div style="margin-top: 15px;">
            <h4>🔍 Search Results for "${query}" (${results.length} found) - Completed in ${searchTime}s</h4>
            ${results.map((result, index) => `
                <div class="search-result-item" style="
                    border: 1px solid #e5e7eb; 
                    border-radius: 8px; 
                    padding: 15px; 
                    margin: 10px 0;
                    background: #f9fafb;
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <strong>Result ${index + 1}</strong>
                        <span style="background: #10b981; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px;">
                            Score: ${(result.similarity_score || 0).toFixed(3)}
                        </span>
                    </div>
                    <p><strong>Chunk ID:</strong> <code>${result.chunk_id}</code></p>
                    <p><strong>Content:</strong></p>
                    <div style="background: white; padding: 10px; border-radius: 4px; border-left: 3px solid #10b981; font-family: monospace; font-size: 14px; max-height: 100px; overflow-y: auto;">
                        ${result.document || 'No content available'}
                    </div>
                    ${result.metadata ? `
                        <details style="margin-top: 8px;">
                            <summary style="cursor: pointer; color: #6b7280;">Metadata</summary>
                            <pre style="background: #f3f4f6; padding: 8px; border-radius: 4px; font-size: 12px; margin-top: 5px;">${JSON.stringify(result.metadata, null, 2)}</pre>
                        </details>
                    ` : ''}
                </div>
            `).join('')}
        </div>
    `;
    
    resultsContainer.innerHTML = resultsHTML;
    console.log('📋 Displayed real search results:', results.length);
}

// Processing pipeline
async function runProcessingPipeline() {
    const steps = [
        { id: 'step-analyze', name: 'Data Analysis', duration: 1000 },
        { id: 'step-preprocess', name: 'Preprocessing', duration: 2000 },
        { id: 'step-chunking', name: 'Chunking', duration: 3000 },
        { id: 'step-embedding', name: 'Embeddings', duration: 4000 },
        { id: 'step-storage', name: 'Vector Storage', duration: 1500 },
        { id: 'step-retrieval', name: 'Retrieval Setup', duration: 1000 }
    ];

    let progress = 0;
    const progressBar = document.getElementById('overall-progress');
    const progressText = document.getElementById('progress-text');

    for (let i = 0; i < steps.length; i++) {
        const step = steps[i];
        
        // Update step status
        updateStepStatus(step.id, 'active');
        progressText.textContent = `Processing: ${step.name}`;

        // Simulate processing with variable duration based on layer complexity
        let duration = step.duration;
        if (currentLayer === 2) duration *= 1.3;
        if (currentLayer === 3) duration *= 1.6;

        await simulateProcessing(duration, (stepProgress) => {
            const totalProgress = ((i + stepProgress) / steps.length) * 100;
            progressBar.style.width = totalProgress + '%';
        });

        // Mark step as completed
        updateStepStatus(step.id, 'completed');
        progress = ((i + 1) / steps.length) * 100;
        progressBar.style.width = progress + '%';
    }

            // Processing completed
            progressText.textContent = 'Processing Complete!';
            await generateResults();
            
            // Change Start Processing button text to Processed
            const processBtn = document.getElementById('process-btn');
            if (processBtn) {
                processBtn.textContent = 'Processed';
                processBtn.disabled = true;
            }
            
            // Show search section after processing is complete
            setTimeout(() => {
                showQuerySection();
            }, 1000);
}

// Simulate processing step
function simulateProcessing(duration, onProgress) {
    return new Promise((resolve) => {
        let elapsed = 0;
        const interval = 50;
        
        const timer = setInterval(() => {
            elapsed += interval;
            const progress = Math.min(elapsed / duration, 1);
            onProgress(progress);
            
            if (progress >= 1) {
                clearInterval(timer);
                resolve();
            }
        }, interval);
    });
}

// Generate mock results based on configuration
async function generateResults() {
    const fileSize = uploadedFile.size;
    const processingTime = Date.now() - processingStartTime;
    
    // Generate realistic results based on layer and file size
    let baseChunks = Math.floor(fileSize / 1000) + Math.floor(Math.random() * 200) + 100;
    let efficiency = 85 + Math.floor(Math.random() * 10);
    let memoryUsage = Math.floor(fileSize / 1024 / 1024 * 2.5) + Math.floor(Math.random() * 100);
    
    // Adjust based on layer complexity
    switch (currentLayer) {
        case 1: // Fast mode
            efficiency += Math.floor(Math.random() * 5);
            break;
        case 2: // Config mode
            baseChunks = Math.floor(baseChunks * 1.2);
            efficiency += Math.floor(Math.random() * 8);
            memoryUsage = Math.floor(memoryUsage * 1.1);
            break;
        case 3: // Deep config
            baseChunks = Math.floor(baseChunks * 1.4);
            efficiency += Math.floor(Math.random() * 12);
            memoryUsage = Math.floor(memoryUsage * 1.3);
            break;
    }

    processedData = {
        chunks: baseChunks,
        embeddings: baseChunks,
        processingTime: Math.floor(processingTime / 1000),
        efficiency: Math.min(efficiency, 98),
        memoryUsage: memoryUsage,
        throughput: Math.floor(baseChunks / (processingTime / 1000)),
        semanticCoherence: 78 + Math.floor(Math.random() * 15),
        chunkDiversity: 65 + Math.floor(Math.random() * 20),
        cpuUsage: 45 + Math.floor(Math.random() * 30),
        gpuUsage: document.getElementById('gpu-acceleration')?.checked ? 60 + Math.floor(Math.random() * 25) : 0
    };

    // Update sidebar stats
    document.getElementById('total-chunks').textContent = processedData.chunks.toLocaleString();
    document.getElementById('processing-time').textContent = processedData.processingTime + 's';
    document.getElementById('memory-usage').textContent = processedData.memoryUsage + 'MB';
}

// Show query section after processing is complete
function showQuerySection() {
    const querySection = document.getElementById('query-section');
    if (querySection) {
        querySection.style.display = 'block';
        // Trigger popup animation
        setTimeout(() => {
            querySection.classList.add('popup-show');
        }, 100);
    }
}

// Perform query search
function performQuery() {
    const queryInput = document.getElementById('query-input');
    const queryResults = document.getElementById('query-results');
    const query = queryInput.value.trim();
    
    if (!query) {
        alert('Please enter a search query');
        return;
    }
    
            // Mock query results (3 chunks)
            const mockResults = [
                {
                    title: "Chunk 1",
                    content: `Customer data analysis showing purchasing patterns across different demographics and regions. This chunk contains detailed information about customer behavior, preferences, and buying trends that can help optimize marketing strategies and product development.`
                },
                {
                    title: "Chunk 2", 
                    content: `Product performance metrics including sales figures, customer satisfaction ratings, and return rates. This data provides insights into which products are performing well and which may need improvement or discontinuation.`
                },
                {
                    title: "Chunk 3",
                    content: `Market trend analysis covering seasonal variations, competitor analysis, and growth projections. This information helps businesses understand market dynamics and make informed strategic decisions.`
                }
            ];
    
    // Display query results
    queryResults.innerHTML = mockResults.map((result, index) => `
        <div class="query-result-item">
            <div class="query-result-title">${result.title}</div>
            <div class="query-result-content">${result.content}</div>
        </div>
    `).join('');
    
    // Ensure action buttons remain visible
    const actionButtons = document.querySelector('.action-buttons');
    if (actionButtons) {
        actionButtons.style.display = 'flex';
        actionButtons.style.visibility = 'visible';
        actionButtons.style.opacity = '1';
        actionButtons.style.position = 'fixed';
        actionButtons.style.bottom = '40px';
        actionButtons.style.right = '40px';
        actionButtons.style.zIndex = '1000';
    }
        }

        // Reset processing function
        function resetProcessing() {
            // Reset all variables
            currentLayer = 1;
            isProcessing = false;
            uploadedFile = null;
            processedData = null;
            processingStartTime = null;
            stepTimers = {};
            stepStartTimes = {};
            fileUploaded = false;
            layerSelected = false; // Reset layer selection state
            
            // Reset UI
            const querySection = document.getElementById('query-section');
            if (querySection) {
                querySection.style.display = 'none';
                querySection.classList.remove('popup-show');
            }
            
            const queryInput = document.getElementById('query-input');
            if (queryInput) {
                queryInput.value = '';
            }
            
            const queryResults = document.getElementById('query-results');
            if (queryResults) {
                queryResults.innerHTML = '';
            }
            
            // Reset Layer 2 UI
            const querySectionLayer2 = document.getElementById('query-section-layer2');
            if (querySectionLayer2) {
                querySectionLayer2.style.display = 'none';
                querySectionLayer2.classList.remove('popup-show');
            }
            
            const queryInputLayer2 = document.getElementById('query-input-layer2');
            if (queryInputLayer2) {
                queryInputLayer2.value = '';
            }
            
            const queryResultsLayer2 = document.getElementById('query-results-layer2');
            if (queryResultsLayer2) {
                queryResultsLayer2.innerHTML = '';
            }
            
            // Reset Layer 2 processing pipeline
            const pipelineSectionLayer2 = document.getElementById('processing-pipeline-section-layer2');
            if (pipelineSectionLayer2) {
                pipelineSectionLayer2.classList.remove('show');
            }
            
            // Reset Layer 1 processing pipeline
            const pipelineSection = document.getElementById('processing-pipeline-section');
            if (pipelineSection) {
                pipelineSection.classList.remove('show');
            }
            
            // Reset sidebar
            const sidebar = document.querySelector('.sidebar');
            if (sidebar) {
                sidebar.classList.remove('show');
            }
            
            // Show layer selection and file upload again
            const layerSelection = document.querySelector('.layer-selection');
            if (layerSelection) {
                layerSelection.style.display = 'flex';
            }
            
            const fileUploadSection = document.querySelector('.file-upload-section');
            if (fileUploadSection) {
                fileUploadSection.style.display = 'block';
            }
            
            // Reset file upload area
            const uploadArea = document.getElementById('file-upload-area');
            if (uploadArea) {
                uploadArea.classList.remove('uploaded');
                uploadArea.innerHTML = `
                    <div class="upload-icon">📁</div>
                    <h3>Upload CSV File</h3>
                    <p>Drag and drop your CSV file here or click to browse</p>
                    <input type="file" id="csvFile" accept=".csv" onchange="handleFileUpload(event)">
                `;
            }
            
            // Reset all step statuses
            const steps = ['step-upload', 'step-analyze', 'step-preprocess', 'step-chunking', 'step-embedding', 'step-storage', 'step-retrieval'];
            steps.forEach(stepId => {
                updateStepStatus(stepId, 'pending');
            });
            
            // Reset Start Processing button text
            const processBtn = document.getElementById('process-btn');
            if (processBtn) {
                processBtn.textContent = 'Start Processing';
                processBtn.disabled = false;
            }
            
            // Reset Layer 2 Start Processing button text
            const processBtnLayer2 = document.getElementById('process-btn-layer2');
            if (processBtnLayer2) {
                processBtnLayer2.textContent = 'Start Processing';
                processBtnLayer2.disabled = false;
            }
            
            // Reset layer selection visual state
            const layerCards = document.querySelectorAll('.layer-card');
            layerCards.forEach(card => {
                card.classList.remove('selected');
            });
            
            // Show main content (title, description, layers)
            const mainContent = document.querySelector('.main-content');
            if (mainContent) {
                mainContent.style.display = 'block';
            }
            
            // Reset all timers
            resetAllTimers();
            
            // Hide action buttons on reset (back to start page)
            const actionButtons = document.querySelector('.action-buttons');
            const actionSection = document.querySelector('.action-section');
            
            if (actionButtons) {
                actionButtons.style.display = 'none';
                actionButtons.style.visibility = 'hidden';
                actionButtons.style.opacity = '0';
            }
            if (actionSection) {
                actionSection.style.display = 'none';
            }
            
            console.log('Application has been reset to starting page');
        }

        // Main reset function with confirmation dialog
        function resetEntireProcess() {
            if (confirm('Do you want to reset the entire process?\n\nThis will:\n• Clear all processing data\n• Remove uploaded files\n• Reset to the starting page\n• Clear all results\n\nClick OK to reset or Cancel to keep current progress.')) {
                resetProcessing();
                alert('Process has been reset successfully!\n\nYou are now back to the starting page with the three layers.');
            }
        }

        // Save config function
        function saveConfig() {
            alert('Configuration saved successfully!');
        }

// Handle Enter key in query input
document.addEventListener('DOMContentLoaded', function() {
    const queryInput = document.getElementById('query-input');
    if (queryInput) {
        queryInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                performQuery();
            }
        });
    }
});

// Show search section
function showSearchSection() {
    const searchSection = document.getElementById('search-section');
    if (searchSection) {
        searchSection.classList.add('show');
    }
}

// Perform search
function performSearch() {
    const searchInput = document.getElementById('search-input');
    const searchResults = document.getElementById('search-results');
    const query = searchInput.value.trim();
    
    if (!query) {
        alert('Please enter a search query');
        return;
    }
    
    // Mock search results (5 chunks)
    const mockResults = [
        {
            title: "Chunk 1",
            content: `Customer data analysis showing purchasing patterns across different demographics and regions. This chunk contains detailed information about customer behavior, preferences, and buying trends that can help optimize marketing strategies and product development.`
        },
        {
            title: "Chunk 2", 
            content: `Product performance metrics including sales figures, customer satisfaction ratings, and return rates. This data provides insights into which products are performing well and which may need improvement or discontinuation.`
        },
        {
            title: "Chunk 3",
            content: `Market trend analysis covering seasonal variations, competitor analysis, and growth projections. This information helps businesses understand market dynamics and make informed strategic decisions.`
        },
        {
            title: "Chunk 4",
            content: `Financial performance data including revenue streams, cost analysis, and profitability metrics. This chunk contains crucial financial information for business planning and investment decisions.`
        },
        {
            title: "Chunk 5",
            content: `Operational efficiency metrics covering production processes, supply chain management, and resource utilization. This data helps identify areas for operational improvement and cost optimization.`
        }
    ];
    
    // Display search results
    searchResults.innerHTML = mockResults.map((result, index) => `
        <div class="search-result-item">
            <div class="search-result-title">${result.title}</div>
            <div class="search-result-content">${result.content}</div>
        </div>
    `).join('');
}

// Handle Enter key in search input
document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('search-input');
    if (searchInput) {
        searchInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                performSearch();
            }
        });
    }
});

// Show results
function showResults() {
    // Results panel removed - no longer needed
}

// Generate sample chunks for display
function generateSampleChunks() {
    const sampleChunks = [
        "Chunk 1: Customer data analysis showing purchasing patterns across different demographics and regions...",
        "Chunk 2: Product performance metrics including sales figures, customer satisfaction ratings, and return rates...",
        "Chunk 3: Market trend analysis covering seasonal variations, competitor analysis, and growth projections...",
        "Chunk 4: Financial performance indicators including revenue growth, profit margins, and cost analysis...",
        "Chunk 5: Operational efficiency metrics covering supply chain optimization and process improvements..."
    ];

    const sampleOutput = document.getElementById('sample-chunks');
    sampleOutput.innerHTML = sampleChunks.map((chunk, index) => 
        `<div style="margin-bottom: 15px; padding: 10px; background: white; border-radius: 8px; border-left: 4px solid #667eea;">
            <strong>Chunk ${index + 1}:</strong><br>
            <span style="color: var(--text-secondary);">${chunk}</span>
        </div>`
    ).join('');
}

// Configuration management
function saveConfiguration() {
    const config = gatherConfiguration();
    const configJson = JSON.stringify(config, null, 2);
    
    // Create download link
    const blob = new Blob([configJson], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `csv_chunking_config_layer_${currentLayer}.json`;
    a.click();
    URL.revokeObjectURL(url);

    alert('Configuration saved successfully!');
}

function exportConfig() {
    saveConfiguration();
}

function importConfig() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                try {
                    const config = JSON.parse(e.target.result);
                    loadConfiguration(config);
                    alert('Configuration loaded successfully!');
                } catch (error) {
                    alert('Invalid configuration file!');
                }
            };
            reader.readAsText(file);
        }
    };
    input.click();
}

function resetConfiguration() {
    if (confirm('Are you sure you want to reset all configuration settings and go back to the first page?')) {
        // Call the main reset function to go back to first page
        resetEntireProcess();
        
        // Reset all form elements to defaults
        document.querySelectorAll('select, input[type="number"], input[type="range"]').forEach(element => {
            if (element.tagName === 'SELECT') {
                element.selectedIndex = 0;
            } else if (element.type === 'number') {
                element.value = element.getAttribute('value') || element.min || 0;
            } else if (element.type === 'range') {
                element.value = element.getAttribute('value') || element.min || 0;
                const valueId = element.id.replace('-range', '-value').replace('range', 'value');
                const valueElement = document.getElementById(valueId);
                if (valueElement) {
                    updateRangeValue(element.id, valueId);
                }
            }
        });

        // Reset checkboxes
        document.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
            checkbox.checked = checkbox.hasAttribute('checked');
        });

        alert('Configuration reset to defaults!');
    }
}

function gatherConfiguration() {
    const config = {
        layer: currentLayer,
        timestamp: new Date().toISOString(),
        preprocessing: {},
        chunking: {},
        embedding: {},
        storage: {},
        retrieval: {},
        performance: {}
    };

    // Gather all form values
    document.querySelectorAll('select, input').forEach(element => {
        if (element.type === 'file') return;
        
        const section = element.id.split('-')[0];
        const key = element.id;
        
        if (element.type === 'checkbox') {
            config[section] = config[section] || {};
            config[section][key] = element.checked;
        } else {
            config[section] = config[section] || {};
            config[section][key] = element.value;
        }
    });

    return config;
}

function loadConfiguration(config) {
    // Load configuration values
    Object.keys(config).forEach(section => {
        if (typeof config[section] === 'object' && config[section] !== null) {
            Object.keys(config[section]).forEach(key => {
                const element = document.getElementById(key);
                if (element) {
                    if (element.type === 'checkbox') {
                        element.checked = config[section][key];
                    } else {
                        element.value = config[section][key];
                        
                        // Update range displays
                        if (element.type === 'range') {
                            const valueId = key + '-value';
                            updateRangeValue(key, valueId);
                        }
                    }
                }
            });
        }
    });

    // Switch to the correct layer
    if (config.layer) {
        selectLayer(config.layer);
    }
}

function exportResults() {
    if (!processedData) {
        alert('No results to export. Please run processing first.');
        return;
    }

    const results = {
        ...processedData,
        timestamp: new Date().toISOString(),
        configuration: gatherConfiguration(),
        fileName: uploadedFile?.name
    };

    const resultsJson = JSON.stringify(results, null, 2);
    const blob = new Blob([resultsJson], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `csv_chunking_results_${new Date().toISOString().slice(0, 19)}.json`;
    a.click();
    URL.revokeObjectURL(url);
}

function startRetrieval() {
    if (!processedData) {
        alert('Please complete processing first before testing retrieval.');
        return;
    }
    
    const query = prompt('Enter a test query for retrieval:');
    if (query) {
        alert(`Testing retrieval with query: "${query}"\n\nThis would return the top ${document.getElementById('top-k-results')?.value || 10} most relevant chunks using ${document.getElementById('similarity-metric')?.value || 'cosine'} similarity.`);
    }
}

// Layer 2 Processing and Query Functions
function startProcessingLayer2() {
    if (isProcessing) return;
    
    isProcessing = true;
    const processBtn = document.getElementById('process-btn-layer2');
    const processingIndicator = document.getElementById('processing-indicator-layer2');
    
    // Reset all timers
    resetAllTimers();
    
    // Hide query section
    const querySection = document.getElementById('query-section-layer2');
    if (querySection) {
        querySection.style.display = 'none';
    }
    
    // Update button state
    processBtn.textContent = 'Processing...';
    processBtn.disabled = true;
    
    // Show processing indicator
    processingIndicator.classList.add('processing');
    
    // Simulate processing steps
    runProcessingPipelineLayer2();
}

function runProcessingPipelineLayer2() {
    const steps = [
        { id: 'step-upload', name: 'File Upload', duration: 1000 },
        { id: 'step-analyze', name: 'Data Analysis', duration: 2000 },
        { id: 'step-preprocess', name: 'Preprocessing', duration: 3000 },
        { id: 'step-chunking', name: 'Chunking', duration: 2500 },
        { id: 'step-embedding', name: 'Embedding', duration: 4000 },
        { id: 'step-storage', name: 'Vector Storage', duration: 2000 },
        { id: 'step-retrieval', name: 'Retrieval Setup', duration: 1500 }
    ];
    
    let currentStep = 0;
    
    function processNextStep() {
        if (currentStep >= steps.length) {
            // All steps completed
            const processBtn = document.getElementById('process-btn-layer2');
            const processingIndicator = document.getElementById('processing-indicator-layer2');
            
            processBtn.textContent = 'Processed';
            processBtn.disabled = true;
            processingIndicator.classList.remove('processing');
            
            // Show query section after processing
            setTimeout(() => {
                showQuerySectionLayer2();
            }, 500);
            
            isProcessing = false;
            return;
        }
        
        const step = steps[currentStep];
        updateStepStatus(step.id, 'active');
        
        setTimeout(() => {
            updateStepStatus(step.id, 'completed');
            currentStep++;
            processNextStep();
        }, step.duration);
    }
    
    processNextStep();
}

function showQuerySectionLayer2() {
    const querySection = document.getElementById('query-section-layer2');
    if (querySection) {
        querySection.style.display = 'block';
        // Trigger popup animation
        setTimeout(() => {
            querySection.classList.add('popup-show');
        }, 100);
    }
}

function performQueryLayer2() {
    const queryInput = document.getElementById('query-input-layer2');
    const queryResults = document.getElementById('query-results-layer2');
    
    if (!queryInput || !queryResults) return;
    
    const query = queryInput.value.trim();
    if (!query) {
        alert('Please enter a query');
        return;
    }
    
    // Generate mock results for Layer 2
    const mockResults = [
        {
            title: "Chunk 1: Data Processing Results",
            content: "This chunk contains processed data from the CSV file with applied preprocessing techniques including null handling and data type conversion as configured in Layer 2."
        },
        {
            title: "Chunk 2: Chunking Analysis",
            content: "Semantic chunking results showing how the data was split into meaningful segments using the configured chunking method and overlap percentage."
        },
        {
            title: "Chunk 3: Vector Embeddings",
            content: "Generated embeddings using the selected model with specified dimensions, ready for similarity search and retrieval operations."
        }
    ];
    
    // Display results
    queryResults.innerHTML = mockResults.map((result, index) => `
        <div class="query-result-item">
            <div class="query-result-title">${result.title}</div>
            <div class="query-result-content">${result.content}</div>
        </div>
    `).join('');
    
    // Ensure action buttons remain visible
    const actionButtons = document.querySelector('.action-buttons');
    if (actionButtons) {
        actionButtons.style.display = 'flex';
        actionButtons.style.position = 'fixed';
        actionButtons.style.bottom = '40px';
        actionButtons.style.right = '40px';
        actionButtons.style.zIndex = '1000';
    }
}

// Initialize the application
function initializeApp() {
    console.log('initializeApp called');
    setupDragDrop();
    
    // Set default range values
    document.querySelectorAll('input[type="range"]').forEach(slider => {
        const valueId = slider.id.replace('-range', '-value').replace(/.*/, match => match + '-value');
        const valueElement = document.getElementById(valueId) || document.getElementById(slider.id + '-value');
        if (valueElement) {
            updateRangeValue(slider.id, valueElement.id);
        }
    });

    // Set initial body class and hide action buttons on start page
    document.body.classList.add(`layer-${currentLayer}`);
    
    // Hide action buttons on start page (before file upload)
    const actionButtons = document.querySelector('.action-buttons');
    const actionSection = document.querySelector('.action-section');
    
    if (actionButtons) {
        actionButtons.style.display = 'none';
        actionButtons.style.visibility = 'hidden';
        actionButtons.style.opacity = '0';
    }
    if (actionSection) {
        actionSection.style.display = 'none';
    }
    
    // Setup mobile sidebar toggle
    setupMobileSidebarToggle();

    console.log('CSV Chunking Optimizer initialized successfully!');
}


// Setup mobile sidebar toggle functionality
function setupMobileSidebarToggle() {
    const toggleButton = document.getElementById('mobile-sidebar-toggle');
    const sidebar = document.querySelector('.sidebar');
    
    if (toggleButton && sidebar) {
        toggleButton.addEventListener('click', function() {
            sidebar.classList.toggle('show');
        });
        
        // Close sidebar when clicking outside on mobile
        document.addEventListener('click', function(event) {
            if (window.innerWidth <= 480) {
                const isClickInsideSidebar = sidebar.contains(event.target);
                const isClickOnToggle = toggleButton.contains(event.target);
                
                if (!isClickInsideSidebar && !isClickOnToggle && sidebar.classList.contains('show')) {
                    sidebar.classList.remove('show');
                }
            }
        });
        
        // Handle window resize
        window.addEventListener('resize', function() {
            if (window.innerWidth > 480) {
                sidebar.classList.add('show');
            } else {
                sidebar.classList.remove('show');
            }
        });
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', initializeApp);

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    if (e.ctrlKey || e.metaKey) {
        switch(e.key) {
            case 's':
                e.preventDefault();
                saveConfiguration();
                break;
            case 'o':
                e.preventDefault();
                importConfig();
                break;
            case 'r':
                if (e.shiftKey) {
                    e.preventDefault();
                    resetConfiguration();
                }
                break;
            case 'Enter':
                if (!isProcessing && uploadedFile) {
                    e.preventDefault();
                    startProcessing();
                }
                break;
        }
    }
});

// ============================================
// DEEP CONFIG FUNCTIONS
// ============================================

// Handle file upload for Deep Config mode
function handleDeepConfigFileUpload(file) {
    console.log('Deep Config file upload:', file.name);
    
    // Read CSV file
    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            const csvText = e.target.result;
            parseCSVData(csvText);
            showCSVPreview();
            showDefaultPreprocessingSection();
        } catch (error) {
            console.error('Error parsing CSV:', error);
            alert('Error parsing CSV file. Please ensure it is a valid CSV format.');
        }
    };
    reader.readAsText(file);
}

// Parse CSV data
function parseCSVData(csvText) {
    const lines = csvText.trim().split('\n');
    const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
    
    csvHeaders = headers;
    csvData = [];
    
    // Parse first 100 rows for preview (limit for performance)
    const dataLines = lines.slice(1, Math.min(101, lines.length));
    
    dataLines.forEach(line => {
        const values = line.split(',').map(v => v.trim().replace(/"/g, ''));
        if (values.length === headers.length) {
            const row = {};
            headers.forEach((header, index) => {
                row[header] = values[index];
            });
            csvData.push(row);
        }
    });
    
    currentPreviewData = csvData.slice(0, 5); // Show first 5 rows
    console.log('CSV parsed:', { headers: csvHeaders, rowCount: csvData.length });
}

// Show CSV preview
function showCSVPreview() {
    const csvPreviewSection = document.getElementById('csv-preview-section');
    const csvHeadersElement = document.getElementById('csv-headers');
    const csvBody = document.getElementById('csv-body');
    
    if (!csvPreviewSection || !currentPreviewData) return;
    
    // Show the preview section
    csvPreviewSection.style.display = 'block';
    
    // Populate headers (show first 3 columns + more indicator)
    const headersToShow = csvHeaders.slice(0, 3);
    const hasMoreColumns = csvHeaders.length > 3;
    
    csvHeadersElement.innerHTML = headersToShow.map(header => `<th>${header}</th>`).join('') +
        (hasMoreColumns ? `<th>... +${csvHeaders.length - 3} more</th>` : '');
    
    // Populate data rows
    csvBody.innerHTML = currentPreviewData.map(row => {
        const cellsToShow = headersToShow.map(header => `<td>${row[header] || ''}</td>`).join('');
        const moreIndicator = hasMoreColumns ? '<td>...</td>' : '';
        return `<tr>${cellsToShow}${moreIndicator}</tr>`;
    }).join('');
    
    // Update progress
    updateStepStatus('step-analyze', 'active');
    setTimeout(() => {
        updateStepStatus('step-analyze', 'completed');
    }, 1000);
}

// Show default preprocessing section
function showDefaultPreprocessingSection() {
    setTimeout(() => {
        const section = document.getElementById('default-preprocessing-section');
        if (section) {
            section.style.display = 'block';
            deepConfigWorkflowStep = 'default-preprocessing';
        }
    }, 1500);
}

// Run default preprocessing
function runDefaultPreprocessing() {
    console.log('Running default preprocessing...');
    
    // Update progress
    updateStepStatus('step-preprocess', 'active');
    
    // Simulate processing steps
    const steps = [
        { id: 'encoding-status', name: 'UTF-8 Encoding', delay: 800 },
        { id: 'headers-status', name: 'Headers Validation', delay: 600 },
        { id: 'whitespace-status', name: 'Whitespace Removal', delay: 700 }
    ];
    
    let currentStep = 0;
    
    function processNextStep() {
        if (currentStep >= steps.length) {
            // All steps completed
            updateStepStatus('step-preprocess', 'completed');
            updatePreviewStatus('After Default Processing');
            showManualPreprocessingSection();
            return;
        }
        
        const step = steps[currentStep];
        const statusElement = document.getElementById(step.id);
        if (statusElement) {
            statusElement.textContent = 'Processing...';
            statusElement.className = 'preprocessing-status processing';
        }
        
        setTimeout(() => {
            if (statusElement) {
                statusElement.textContent = 'Completed';
                statusElement.className = 'preprocessing-status completed';
            }
            currentStep++;
            processNextStep();
        }, step.delay);
    }
    
    // Disable button
    const runBtn = document.getElementById('run-default-btn');
    if (runBtn) {
        runBtn.disabled = true;
        runBtn.textContent = 'Processing...';
    }
    
    processNextStep();
}

// Show manual preprocessing section
function showManualPreprocessingSection() {
    setTimeout(() => {
        const section = document.getElementById('manual-preprocessing-section');
        if (section) {
            section.style.display = 'block';
            deepConfigWorkflowStep = 'manual-preprocessing';
            showDataTypeConversionStep();
        }
    }, 1000);
}

// Show data type conversion step
function showDataTypeConversionStep() {
    const step = document.getElementById('datatype-conversion-step');
    if (step) {
        step.style.display = 'block';
        populateColumnSelects();
        manualPreprocessingStep = 'datatype';
    }
}

// Populate column dropdowns
function populateColumnSelects() {
    const datatypeSelect = document.getElementById('datatype-column-select');
    const nullSelect = document.getElementById('null-column-select');
    
    if (datatypeSelect && csvHeaders.length > 0) {
        datatypeSelect.innerHTML = '<option value="">Select a column...</option>' +
            csvHeaders.map((header, index) => 
                `<option value="${header}">'${header}' (current: ${getColumnType(header)})</option>`
            ).join('');
    }
    
    if (nullSelect && csvHeaders.length > 0) {
        nullSelect.innerHTML = '<option value="">Select a column...</option>' +
            csvHeaders.map(header => 
                `<option value="${header}">${header} (${getNullCount(header)} nulls)</option>`
            ).join('');
    }
}

// Get column data type (mock implementation)
function getColumnType(columnName) {
    if (!csvData || csvData.length === 0) return 'object';
    
    const sampleValue = csvData[0][columnName];
    if (!isNaN(sampleValue) && sampleValue !== '') return 'numeric';
    if (sampleValue === 'true' || sampleValue === 'false') return 'bool';
    return 'object';
}

// Get null count for column (mock implementation)
function getNullCount(columnName) {
    if (!csvData) return 0;
    
    return csvData.filter(row => !row[columnName] || row[columnName].trim() === '').length;
}

// Apply data type conversion
function applyDataTypeConversion() {
    const column = document.getElementById('datatype-column-select').value;
    const strategy = document.querySelector('input[name="datatype-strategy"]:checked').value;
    
    if (!column) {
        alert('Please select a column to convert.');
        return;
    }
    
    console.log(`Applying ${strategy} conversion to column: ${column}`);
    
    // Simulate processing
    setTimeout(() => {
        updatePreviewStatus('After Data Type Conversion');
        showNullHandlingStep();
    }, 1000);
}

// Show null handling step
function showNullHandlingStep() {
    const currentStep = document.getElementById('datatype-conversion-step');
    const nextStep = document.getElementById('null-handling-step');
    
    if (currentStep) currentStep.style.display = 'none';
    if (nextStep) {
        nextStep.style.display = 'block';
        manualPreprocessingStep = 'null';
        
        // Show custom value group if custom is selected
        const customRadio = document.querySelector('input[name="null-strategy"][value="custom"]');
        const customGroup = document.getElementById('custom-value-group');
        
        document.querySelectorAll('input[name="null-strategy"]').forEach(radio => {
            radio.addEventListener('change', function() {
                if (customGroup) {
                    customGroup.style.display = this.value === 'custom' ? 'block' : 'none';
                }
            });
        });
    }
}

// Apply null handling
function applyNullHandling() {
    const column = document.getElementById('null-column-select').value;
    const strategy = document.querySelector('input[name="null-strategy"]:checked').value;
    
    if (!column) {
        alert('Please select a column to handle nulls.');
        return;
    }
    
    console.log(`Applying ${strategy} null handling to column: ${column}`);
    
    // Simulate processing
    setTimeout(() => {
        updatePreviewStatus('After Null Handling');
        showDuplicateHandlingStep();
    }, 1000);
}

// Show duplicate handling step
function showDuplicateHandlingStep() {
    const currentStep = document.getElementById('null-handling-step');
    const nextStep = document.getElementById('duplicate-handling-step');
    
    if (currentStep) currentStep.style.display = 'none';
    if (nextStep) {
        nextStep.style.display = 'block';
        manualPreprocessingStep = 'duplicate';
    }
}

// Apply duplicate handling
function applyDuplicateHandling() {
    const strategy = document.querySelector('input[name="duplicate-strategy"]:checked').value;
    
    console.log(`Applying ${strategy} duplicate handling`);
    
    // Simulate processing
    setTimeout(() => {
        updatePreviewStatus('After Duplicate Handling');
        showTextProcessingStep();
    }, 1000);
}

// Show text processing step
function showTextProcessingStep() {
    const currentStep = document.getElementById('duplicate-handling-step');
    const nextStep = document.getElementById('text-processing-step');
    
    if (currentStep) currentStep.style.display = 'none';
    if (nextStep) {
        nextStep.style.display = 'block';
        manualPreprocessingStep = 'text';
    }
}

// Apply text processing
function applyTextProcessing() {
    const stopwordStrategy = document.querySelector('input[name="stopword-strategy"]:checked').value;
    const normalizationStrategy = document.querySelector('input[name="normalization-strategy"]:checked').value;
    
    console.log(`Applying text processing: stopwords=${stopwordStrategy}, normalization=${normalizationStrategy}`);
    
    // Simulate processing
    setTimeout(() => {
        updatePreviewStatus('After Text Processing');
        showDownloadPreprocessedSection();
        showMetadataHandlingSection();
    }, 1000);
}

// Show download preprocessed section
function showDownloadPreprocessedSection() {
    const currentStep = document.getElementById('text-processing-step');
    const downloadSection = document.getElementById('download-preprocessed-section');
    
    if (currentStep) currentStep.style.display = 'none';
    if (downloadSection) downloadSection.style.display = 'block';
}

// Download preprocessed CSV
function downloadPreprocessedCSV() {
    console.log('Downloading preprocessed CSV...');
    alert('Preprocessed CSV would be downloaded as ZIP file.');
}

// Show metadata handling section
function showMetadataHandlingSection() {
    setTimeout(() => {
        const section = document.getElementById('metadata-handling-section');
        if (section) {
            section.style.display = 'block';
            deepConfigWorkflowStep = 'metadata';
            setupMetadataHandling();
        }
    }, 1500);
}

// Setup metadata handling
function setupMetadataHandling() {
    const numericSelect = document.getElementById('numeric-columns-select');
    const categoricalSelect = document.getElementById('categorical-columns-select');
    
    if (numericSelect && csvHeaders.length > 0) {
        numericSelect.innerHTML = csvHeaders
            .filter(header => getColumnType(header) === 'numeric')
            .map(header => `<option value="${header}">${header}</option>`)
            .join('');
        
        // Add event listener for selection
        numericSelect.addEventListener('change', updateSelectedColumns);
    }
    
    if (categoricalSelect && csvHeaders.length > 0) {
        categoricalSelect.innerHTML = csvHeaders
            .filter(header => getColumnType(header) === 'object')
            .map(header => `<option value="${header}">${header}</option>`)
            .join('');
        
        // Add event listener for selection
        categoricalSelect.addEventListener('change', updateSelectedColumns);
    }
    
    // Show chunking section after metadata
    setTimeout(() => {
        showChunkingSection();
    }, 2000);
}

// Update selected columns display
function updateSelectedColumns() {
    const numericSelect = document.getElementById('numeric-columns-select');
    const categoricalSelect = document.getElementById('categorical-columns-select');
    const numericDisplay = document.getElementById('selected-numeric-columns');
    const categoricalDisplay = document.getElementById('selected-categorical-columns');
    
    if (numericSelect && numericDisplay) {
        selectedNumericColumns = Array.from(numericSelect.selectedOptions).map(option => option.value);
        numericDisplay.innerHTML = selectedNumericColumns
            .map(col => `<span class="selected-column-tag">${col}</span>`)
            .join('');
    }
    
    if (categoricalSelect && categoricalDisplay) {
        selectedCategoricalColumns = Array.from(categoricalSelect.selectedOptions).map(option => option.value);
        categoricalDisplay.innerHTML = selectedCategoricalColumns
            .map(col => `<span class="selected-column-tag">${col}</span>`)
            .join('');
    }
}

// Show chunking section
function showChunkingSection() {
    const section = document.getElementById('chunking-section');
    if (section) {
        section.style.display = 'block';
        deepConfigWorkflowStep = 'chunking';
        setupChunkingMethodListeners();
    }
}

// Setup chunking method listeners
function setupChunkingMethodListeners() {
    const methodRadios = document.querySelectorAll('input[name="chunking-method"]');
    methodRadios.forEach(radio => {
        radio.addEventListener('change', function() {
            showChunkingParameters(this.value);
        });
    });
}

// Show chunking parameters based on method
function showChunkingParameters(method) {
    // Hide all parameter sections
    const parameterSections = document.querySelectorAll('.chunking-parameters');
    parameterSections.forEach(section => section.style.display = 'none');
    
    // Show selected parameter section
    const selectedSection = document.getElementById(`${method}-parameters`);
    if (selectedSection) {
        selectedSection.style.display = 'block';
    }
}

// Start chunking
function startChunking() {
    const processSection = document.getElementById('chunking-process-section');
    if (processSection) {
        processSection.style.display = 'block';
        updateStepStatus('step-chunking', 'active');
    }
}

// Apply chunking method
function applyChunkingMethod() {
    const method = document.querySelector('input[name="chunking-method"]:checked').value;
    
    console.log(`Applying ${method} chunking method`);
    
    // Simulate chunking process
    setTimeout(() => {
        updateStepStatus('step-chunking', 'completed');
        showChunkingCompleteSection();
        chunkingComplete = true;
    }, 2000);
}

// Show chunking complete section
function showChunkingCompleteSection() {
    const completeSection = document.getElementById('chunking-complete-section');
    if (completeSection) {
        completeSection.style.display = 'block';
        
        // Show embeddings section after chunking
        setTimeout(() => {
            showEmbeddingsSection();
        }, 1000);
    }
}

// Download chunks as ZIP
function downloadChunksAsZip() {
    console.log('Downloading chunks as ZIP...');
    alert('All chunks would be downloaded as ZIP file.');
}

// Start new chunking
function startNewChunking() {
    const completeSection = document.getElementById('chunking-complete-section');
    const processSection = document.getElementById('chunking-process-section');
    
    if (completeSection) completeSection.style.display = 'none';
    if (processSection) processSection.style.display = 'block';
    
    chunkingComplete = false;
    updateStepStatus('step-chunking', 'pending');
}

// Show embeddings section
function showEmbeddingsSection() {
    const section = document.getElementById('embeddings-section');
    if (section) {
        section.style.display = 'block';
        deepConfigWorkflowStep = 'embeddings';
    }
}

// Generate embeddings
function generateEmbeddings() {
    const model = document.querySelector('input[name="embedding-model"]:checked').value;
    const batchSize = document.getElementById('embedding-batch-size').value;
    
    console.log(`Generating embeddings with model: ${model}, batch size: ${batchSize}`);
    
    updateStepStatus('step-embedding', 'active');
    
    // Simulate embedding generation
    setTimeout(() => {
        updateStepStatus('step-embedding', 'completed');
        showEmbeddingsCompleteSection();
        embeddingsComplete = true;
    }, 3000);
}

// Show embeddings complete section
function showEmbeddingsCompleteSection() {
    const completeSection = document.getElementById('embeddings-complete-section');
    if (completeSection) {
        completeSection.style.display = 'block';
        
        // Show storing section after embeddings
        setTimeout(() => {
            showStoringSection();
        }, 1000);
    }
}

// Download embeddings as JSON
function downloadEmbeddingsAsJson() {
    console.log('Downloading embeddings as JSON...');
    alert('Embeddings would be downloaded as JSON file.');
}

// Start new embeddings process
function startNewEmbeddingsProcess() {
    const completeSection = document.getElementById('embeddings-complete-section');
    if (completeSection) completeSection.style.display = 'none';
    
    embeddingsComplete = false;
    updateStepStatus('step-embedding', 'pending');
}

// Show storing section
function showStoringSection() {
    const section = document.getElementById('storing-section');
    if (section) {
        section.style.display = 'block';
        deepConfigWorkflowStep = 'storing';
    }
}

// Start storing process
function startStoringProcess() {
    const model = document.querySelector('input[name="storage-model"]:checked').value;
    const persistDir = document.getElementById('persist-directory').value;
    const collectionName = document.getElementById('collection-name').value;
    
    console.log(`Starting storage process: model=${model}, dir=${persistDir}, collection=${collectionName}`);
    
    updateStepStatus('step-storage', 'active');
    
    // Simulate storing process
    setTimeout(() => {
        updateStepStatus('step-storage', 'completed');
        showRetrievalSection();
    }, 2000);
}

// Show retrieval section
function showRetrievalSection() {
    const section = document.getElementById('retrieval-section');
    if (section) {
        section.style.display = 'block';
        deepConfigWorkflowStep = 'retrieval';
    }
}

// Complete process
function completeProcess() {
    updateStepStatus('step-retrieval', 'active');
    
    setTimeout(() => {
        updateStepStatus('step-retrieval', 'completed');
        showQueryInterface();
    }, 1000);
}

// Show query interface
function showQueryInterface() {
    const querySection = document.getElementById('query-interface-section');
    if (querySection) {
        querySection.style.display = 'block';
        
        // Setup enter key listener for deep config query
        const queryInput = document.getElementById('deep-config-query-input');
        if (queryInput) {
            queryInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    performDeepConfigQuery();
                }
            });
        }
    }
}

// Perform deep config query
function performDeepConfigQuery() {
    const queryInput = document.getElementById('deep-config-query-input');
    const queryResults = document.getElementById('deep-config-query-results');
    const query = queryInput.value.trim();
    
    if (!query) {
        alert('Please enter a search query');
        return;
    }
    
    console.log('Performing deep config query:', query);
    
    // Mock query results (top 5 chunks)
    const mockResults = [
        {
            title: "Chunk 1: Processed Data Results",
            content: `Advanced processed data from your CSV file with applied preprocessing, chunking, and embedding techniques. This chunk represents optimized data segments ready for retrieval.`
        },
        {
            title: "Chunk 2: Metadata-Enhanced Content", 
            content: `This chunk contains your data enhanced with selected metadata from ${selectedNumericColumns.length} numeric and ${selectedCategoricalColumns.length} categorical columns.`
        },
        {
            title: "Chunk 3: Semantic Analysis",
            content: `Semantically processed chunk using the selected embedding model with optimized parameters for your specific dataset and use case.`
        },
        {
            title: "Chunk 4: Structured Information",
            content: `Structured data chunk with applied text processing, null handling, and duplicate removal based on your configuration choices.`
        },
        {
            title: "Chunk 5: Query-Optimized Content",
            content: `Final processed chunk optimized for retrieval queries using the configured similarity metrics and storage backend.`
        }
    ];
    
    // Display results
    queryResults.innerHTML = mockResults.map((result, index) => `
        <div class="query-result-item">
            <div class="query-result-title">${result.title}</div>
            <div class="query-result-content">${result.content}</div>
        </div>
    `).join('');
}

// Update preview status
function updatePreviewStatus(status) {
    const statusElement = document.getElementById('preview-status');
    if (statusElement) {
        statusElement.textContent = status;
    }
}
