import React, { useEffect, useMemo, useState } from "react";

const MAX_SIZE_MB = 5;
const allowedTypes = ["image/png", "image/jpeg", "image/jpg"];

const ImageUpload = ({ onSubmit, error }) => {
  const [file, setFile] = useState(null);
  const [localError, setLocalError] = useState(null);

  const previewUrl = useMemo(() => {
    if (!file) {
      return null;
    }
    return URL.createObjectURL(file);
  }, [file]);

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const validateFile = (selectedFile) => {
    if (!selectedFile) {
      return "Please upload an image file.";
    }
    if (!allowedTypes.includes(selectedFile.type)) {
      return "Only PNG or JPEG images are accepted.";
    }
    if (selectedFile.size / (1024 * 1024) > MAX_SIZE_MB) {
      return `File must be smaller than ${MAX_SIZE_MB}MB.`;
    }
    return null;
  };

  const handleFile = (selectedFile) => {
    const message = validateFile(selectedFile);
    setLocalError(message);
    if (!message) {
      setFile(selectedFile);
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const droppedFile = event.dataTransfer.files[0];
    handleFile(droppedFile);
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    const message = validateFile(file);
    setLocalError(message);
    if (!message) {
      onSubmit(file);
    }
  };

  return (
    <div className="card">
      <h2>Upload Imaging</h2>
      <p className="muted">
        Drag and drop a diagnostic image or browse to upload. Supported formats: PNG, JPG.
      </p>
      <form onSubmit={handleSubmit} className="form">
        <div
          className="dropzone"
          onDrop={handleDrop}
          onDragOver={handleDragOver}
        >
          <input
            type="file"
            accept="image/png, image/jpeg"
            onChange={(event) => handleFile(event.target.files[0])}
          />
          {previewUrl ? (
            <img src={previewUrl} alt="Uploaded preview" className="preview" />
          ) : (
            <p>Drop your image here or click to browse.</p>
          )}
        </div>

        {(localError || error) && (
          <div className="alert">
            {localError || (typeof error === "string" ? error : "Upload failed.")}
          </div>
        )}

        <button type="submit" className="button" disabled={!file}>
          Analyze Image
        </button>
      </form>
    </div>
  );
};

export default ImageUpload;
