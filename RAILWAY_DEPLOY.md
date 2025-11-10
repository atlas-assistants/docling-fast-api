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
- `ENV=production` (optional)
- `OMP_NUM_THREADS=2` (optional, for performance tuning)

### Build Time
The first build will take longer because it needs to:
- Download PyTorch CPU version
- Download Docling models
- Download EasyOCR models

Subsequent builds will be faster due to Docker layer caching.

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

If you encounter memory issues:
1. Increase Railway plan memory
2. Reduce `OMP_NUM_THREADS` environment variable
3. Consider processing smaller documents

If build fails:
- Check Railway build logs
- Ensure all dependencies are in `pyproject.toml`
- Verify Dockerfile syntax

