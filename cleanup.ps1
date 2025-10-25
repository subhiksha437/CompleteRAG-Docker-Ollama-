# Complete cleanup
Write-Host "
🗑️  Removing containers and volumes..." -ForegroundColor Yellow
docker-compose down -v
Write-Host "
⚠️  Removing local data..." -ForegroundColor Red
Remove-Item -Recurse -Force chroma_db, chat_history, exports -ErrorAction SilentlyContinue
Write-Host "✅ Cleanup complete!" -ForegroundColor Green
