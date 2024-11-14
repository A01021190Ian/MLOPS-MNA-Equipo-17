# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy files to container
COPY tumor_model.pkl /app/
COPY TCGA_GBM_LGG_Mutations_clean.csv /app/
COPY main.py /app/

# Install dependencies
RUN pip install fastapi uvicorn scikit-learn pydantic pandas

# Expose port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]