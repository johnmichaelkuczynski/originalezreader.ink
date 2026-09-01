/**
 * Azure Text-to-Speech Integration
 * This file handles the TTS UI interactions and API calls
 */

document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const createAudiobookBtn = document.getElementById('create-audiobook-btn');
    const convertToSpeechBtn = document.getElementById('convertToSpeechBtn');
    const genderSelect = document.getElementById('gender-select');
    const languageSelect = document.getElementById('language-select');
    const useReducedLengthCheckbox = document.getElementById('use-reduced-length');
    const ttsControls = document.getElementById('tts-controls');
    const ttsProcessing = document.getElementById('tts-processing');
    const ttsResult = document.getElementById('tts-result');
    const ttsError = document.getElementById('tts-error');
    const ttsAudioPlayer = document.getElementById('tts-audio-player');
    const ttsDownloadLink = document.getElementById('tts-download-link');
    const ttsLanguageInfo = document.getElementById('tts-language-info');
    const ttsErrorMessage = document.getElementById('tts-error-message');
    const ttsResetBtn = document.getElementById('tts-reset-btn');
    const ttsRetryBtn = document.getElementById('tts-retry-btn');
    
    // Text-to-Speech functionality with Azure
    if (createAudiobookBtn) {
        createAudiobookBtn.addEventListener('click', function() {
            // Get text from the output or input area
            const textToConvert = document.getElementById('outputText').value || document.getElementById('inputText').value;
            
            if (!textToConvert) {
                showNotification('No text available to convert to speech.', 'warning');
                return;
            }
            
            // Show processing view
            ttsControls.classList.add('d-none');
            ttsProcessing.classList.remove('d-none');
            ttsResult.classList.add('d-none');
            ttsError.classList.add('d-none');
            
            // Get options
            const gender = genderSelect.value;
            const language = languageSelect.value; // Empty string for auto-detect
            const useReducedLength = useReducedLengthCheckbox.checked;
            
            // Call the backend API
            fetch('/create_audiobook', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: textToConvert,
                    gender: gender,
                    language: language || null, // Convert empty string to null
                    use_reduced_length: useReducedLength
                }),
            })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(data => {
                        throw new Error(data.error || 'An error occurred while creating the audiobook.');
                    });
                }
                return response.json();
            })
            .then(data => {
                // Success - show the audio player
                ttsProcessing.classList.add('d-none');
                ttsResult.classList.remove('d-none');
                
                // Set audio source
                ttsAudioPlayer.src = data.audio_url;
                
                // Set download link
                ttsDownloadLink.href = data.download_url;
                
                // Show language info
                let langInfo = `Audio created using ${data.narrator}`;
                if (data.language_name) {
                    langInfo += ` in ${data.language_name}`;
                }
                ttsLanguageInfo.textContent = langInfo;
                
                // Play the audio
                ttsAudioPlayer.load();
                // Don't auto-play - let the user decide when to play
            })
            .catch(error => {
                console.error('Error creating audiobook:', error);
                
                // Show error view
                ttsProcessing.classList.add('d-none');
                ttsError.classList.remove('d-none');
                
                // Set error message
                ttsErrorMessage.textContent = error.message || 'There was an error generating your audio.';
            });
        });
    }
    
    // "Convert to Speech" button in the main interface
    if (convertToSpeechBtn) {
        convertToSpeechBtn.addEventListener('click', function() {
            // Scroll to the TTS section
            const ttsSection = document.querySelector('.card-header.bg-primary.text-white');
            if (ttsSection) {
                ttsSection.scrollIntoView({ behavior: 'smooth' });
                
                // Optional: highlight the generate button
                setTimeout(() => {
                    if (createAudiobookBtn) {
                        createAudiobookBtn.classList.add('btn-pulse');
                        setTimeout(() => {
                            createAudiobookBtn.classList.remove('btn-pulse');
                        }, 1500);
                    }
                }, 500);
            }
        });
    }
    
    // Reset button (create another)
    if (ttsResetBtn) {
        ttsResetBtn.addEventListener('click', function() {
            // Show controls, hide results
            ttsControls.classList.remove('d-none');
            ttsResult.classList.add('d-none');
            ttsError.classList.add('d-none');
            
            // Stop audio if playing
            ttsAudioPlayer.pause();
            ttsAudioPlayer.currentTime = 0;
        });
    }
    
    // Retry button
    if (ttsRetryBtn) {
        ttsRetryBtn.addEventListener('click', function() {
            // Show controls, hide error
            ttsControls.classList.remove('d-none');
            ttsError.classList.add('d-none');
        });
    }
});

// Helper function to show notifications
function showNotification(message, type = 'info') {
    const alertBox = document.createElement('div');
    alertBox.className = `alert alert-${type} alert-dismissible fade show fixed-top mx-auto mt-3`;
    alertBox.style.maxWidth = '500px';
    alertBox.style.zIndex = '9999';
    alertBox.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    document.body.appendChild(alertBox);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        alertBox.classList.remove('show');
        setTimeout(() => alertBox.remove(), 300);
    }, 5000);
}

// Add a little CSS for the pulse effect
const style = document.createElement('style');
style.textContent = `
.btn-pulse {
    animation: pulse 1.5s ease-in-out;
}

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(0, 123, 255, 0.7); }
    70% { box-shadow: 0 0 0 15px rgba(0, 123, 255, 0); }
    100% { box-shadow: 0 0 0 0 rgba(0, 123, 255, 0); }
}
`;
document.head.appendChild(style);