# Deploying to Railway

## Quick Deploy Steps

1. **Push your code to GitHub** (or connect directly via Railway CLI)

2. **Create a new Railway project:**
   - Go to [railway.app](https://railway.app)
   - Click "New Project"
   - Select "Deploy from GitHub repo" (or use Railway CLI)

3. **Railway will automatically:**
   - Detect the Dockerfile
   - Build the Docker image
   - Deploy the application

## Important Notes

### Memory Requirements
This application uses Docling and PyTorch, which can be memory-intensive. Railway's default plan may not be sufficient. Consider:
- **Hobby Plan**: 512MB RAM (may work for small documents)
- **Pro Plan**: 8GB RAM (recommended for production)

### Environment Variables
Railway automatically provides:
- `PORT` - The port your app should listen on (automatically configured)

You can add custom environment variables in Railway dashboard if needed:
- `OMP_NUM_THREADS=2` (optional, for performance tuning)
- `MALLOC_ARENA_MAX=2` (optional, for memory optimization)

### Build Time
**Build time is now optimized!** The Docker build should complete in 2-5 minutes because:
- PyTorch is installed during build (cached for subsequent builds)
- Models are downloaded at runtime on first use (not during build)

**First API call will be slower** (30-60 seconds) because models download on first use:
- Docling pipeline models (~500MB)
- EasyOCR language models (as needed)

Subsequent API calls will be fast since models are cached in memory.

### Deployment Methods

Railway supports multiple deployment methods:

1. **Dockerfile** (recommended - already configured)
   - Railway will automatically detect and use the Dockerfile

2. **Procfile** (alternative)
   - If you prefer not to use Docker, Railway can use the Procfile

3. **Nixpacks** (automatic fallback)
   - Railway can auto-detect Python and build automatically

## Monitoring

After deployment:
- Railway provides logs in the dashboard
- API will be available at: `https://your-app-name.up.railway.app`
- API docs: `https://your-app-name.up.railway.app/docs`

## Troubleshooting

### Build Getting Stuck at "Exporting to Docker Image"
**SOLVED!** The Dockerfile has been optimized to avoid this issue:
- Models are no longer pre-downloaded during build
- Image size is much smaller
- Build completes in 2-5 minutes instead of timing out

### Memory Issues
If you encounter memory issues during API calls:
1. Increase Railway plan memory (recommend 2GB+)
2. Reduce `OMP_NUM_THREADS` environment variable
3. Consider processing smaller documents

### Build Failures
If build still fails:
- Check Railway build logs for specific errors
- Ensure all dependencies are in `pyproject.toml`
- Verify Dockerfile syntax
- Check if Railway has sufficient disk space during build

### First Request Timeout
If first API request times out:
- This is normal - models are downloading (~500MB)
- Increase Railway timeout settings if needed
- Subsequent requests will be fast

