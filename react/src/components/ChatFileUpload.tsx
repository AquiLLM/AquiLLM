import React, { useState, ChangeEvent, useEffect } from 'react';

export interface ProcessedFile {
  originalFile: File;
  name: string;
  size: number;
  type: string;
  base64: string;
  isProcessing: boolean;
  error?: string;
}

interface FileUploadProps {
    onFileUpload: (files: ProcessedFile[]) => void;
}

const ChatFileUpload: React.FC<FileUploadProps> = ({ onFileUpload }) => {
    const [processedFiles, setProcessedFiles] = useState<ProcessedFile[]>([]);
    const [isProcessing, setIsProcessing] = useState<boolean>(false);

    useEffect(() => {
        const completedFiles = processedFiles.filter(file => !file.isProcessing);
        onFileUpload(completedFiles);
    }, [processedFiles, onFileUpload]);    

    const convertFileToBase64 = (file: File): Promise<string> => {
        return new Promise((resolve, reject) => {
        const reader = new FileReader();
        
        reader.onload = () => {
            // Extract base64 data from the data URL
            const result = reader.result as string;
            const base64String = result.split(',')[1];
            resolve(base64String);
        };
        
        reader.onerror = () => {
            reject(new Error('Failed to read file'));
        };
        
        reader.readAsDataURL(file);
        });
    };

    const handleFileUpload = async (event: ChangeEvent<HTMLInputElement>) => {
        const fileList = event.target.files;
        if (!fileList || fileList.length === 0) return;

        setIsProcessing(true);
        
    // Create placeholder objects for all files with isProcessing = true
    const newFiles: ProcessedFile[] = Array.from(fileList).map(file => ({
      originalFile: file,
      name: file.name,
      size: file.size,
      type: file.type,
      base64: '',
      isProcessing: true
    }));
    
    // Add the new files to our state
    setProcessedFiles(prev => [...prev, ...newFiles]);
    
    // Process each file and update its state once complete
    const processPromises = Array.from(fileList).map(async (file, index) => {
      try {
        const base64Data = await convertFileToBase64(file);
        
        // Update just this file's entry in the state
        setProcessedFiles(currentFiles => {
          const newFilesArray = [...currentFiles];
          const fileIndex = newFilesArray.findIndex(
            f => f.name === file.name && f.size === file.size && f.isProcessing
          );
          
          if (fileIndex !== -1) {
            newFilesArray[fileIndex] = {
              ...newFilesArray[fileIndex],
              base64: base64Data,
              isProcessing: false
            };
          }
          
          return newFilesArray;
        });
        
      } catch (error) {
        // Handle errors per file
        setProcessedFiles(currentFiles => {
          const newFilesArray = [...currentFiles];
          const fileIndex = newFilesArray.findIndex(
            f => f.name === file.name && f.size === file.size && f.isProcessing
          );
          
          if (fileIndex !== -1) {
            newFilesArray[fileIndex] = {
              ...newFilesArray[fileIndex],
              isProcessing: false,
              error: 'Failed to convert file'
            };
          }
          
          return newFilesArray;
        });
      }
    });
    
    // When all files are processed, clear the input
    await Promise.all(processPromises);
    setIsProcessing(false);
    
    // Reset the file input so the same files can be selected again if needed
    if (event.target.value) {
      event.target.value = '';
    }
  };

  const removeFile = (index: number) => {
    setProcessedFiles(files => files.filter((_, i) => i !== index));
  };

  const clearAllFiles = () => {
    setProcessedFiles([]);
  };

  // Format file size in a human-readable way
  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} bytes`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  // Display a truncated version of the base64 string
  const truncateBase64 = (base64: string, length: number = 50): string => {
    if (!base64) return '';
    if (base64.length <= length) return base64;
    return `${base64.substring(0, length)}...`;
  };

  return (
    <div className="multi-file-converter">
      
      <div className="upload-section">
        <label htmlFor="file-upload" className="upload-button">
          {isProcessing ? 'Processing...' : 'Choose Files'}
        </label>
        <input 
          id="file-upload"
          type="file"
          multiple
          onChange={handleFileUpload}
          style={{ display: 'none' }}
          disabled={isProcessing}
        />
      </div>

      <div className="files-list">
        {processedFiles.map((file, index) => (
          <div key={`${file.name}-${index}`} className="file-item">
            <div className="file-header">
              <span className="file-name">{file.name}</span>
              <button 
                onClick={() => removeFile(index)}
                className="remove-button"
                disabled={file.isProcessing}
              >
                ✕
              </button>
            </div>
            
            <div className="file-details">
              <p><strong>Size:</strong> {formatFileSize(file.size)}</p>
              
              {file.isProcessing ? (
                <p className="processing">Processing...</p>
              ) : file.error ? (
                <p className="error">{file.error}</p>
              ) : (
                <>
                  <div className="base64-preview">
                    <div className="base64-content">{truncateBase64(file.base64)}</div>
                  </div>
                  
                </>
              )}
            </div>
          </div>
        ))}
      </div>

    </div>
  );
};

export default ChatFileUpload;