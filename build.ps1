# Build Docker images
Write-Host "
🔨 Building Docker images..." -ForegroundColor Cyan
docker-compose build
Write-Host "
✅ Build complete!" -ForegroundColor Green
