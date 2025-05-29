import React, { useState, useEffect } from 'react';
import { 
  Typography, Paper, Grid, Card, CardContent, 
  CardHeader, CardActions, Button, Box, CircularProgress 
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

export default function Dashboard() {
  const navigate = useNavigate();
  const [projectStats, setProjectStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await axios.get('/api/stats');
        setProjectStats(response.data);
        setLoading(false);
      } catch (err) {
        console.error('Error fetching project stats:', err);
        setError('Failed to load project statistics. Server might not be running.');
        setLoading(false);
        
        // Fallback data for demonstration
        setProjectStats({
          totalPapers: 639,
          totalClusters: 5,
          categoriesCount: 22,
          clusterSizes: {
            0: 417,
            1: 85, 
            2: 60,
            3: 48,
            4: 29
          },
          processingStatus: 'completed',
          lastRun: '2025-05-28 23:05:01'
        });
      }
    };

    fetchStats();
  }, []);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="80vh">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <div>
      <Typography variant="h4" gutterBottom>
        Academic Paper Clustering Dashboard
      </Typography>
      
      {error && (
        <Paper sx={{ p: 2, mb: 3, bgcolor: '#fff3f3' }}>
          <Typography color="error">{error}</Typography>
        </Paper>
      )}
      
      <Grid container spacing={3}>
        {/* Project Overview */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader title="Project Overview" />
            <CardContent>
              <Typography variant="body1" paragraph>
                This application helps you cluster academic papers from ArXiv based on their content.
                It uses PySpark for distributed data processing and machine learning to identify research areas.
              </Typography>
              <Typography variant="body1">
                Start by collecting data from ArXiv, then perform clustering analysis to discover patterns
                in research papers.
              </Typography>
            </CardContent>
            <CardActions>
              <Button 
                variant="contained" 
                color="primary"
                onClick={() => navigate('/data-collection')}
              >
                Start New Analysis
              </Button>
            </CardActions>
          </Card>
        </Grid>

        {/* Project Statistics */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader title="Current Statistics" />
            <CardContent>
              {projectStats ? (
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="subtitle1">Total Papers</Typography>
                    <Typography variant="h4">{projectStats.totalPapers}</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="subtitle1">Number of Clusters</Typography>
                    <Typography variant="h4">{projectStats.totalClusters}</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="subtitle1">Categories Count</Typography>
                    <Typography variant="h4">{projectStats.categoriesCount}</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="subtitle1">Processing Status</Typography>
                    <Typography variant="h6" sx={{ 
                      color: projectStats.processingStatus === 'completed' ? 'green' : 'orange'
                    }}>
                      {projectStats.processingStatus === 'completed' ? 'Completed' : 'In Progress'}
                    </Typography>
                  </Grid>
                  <Grid item xs={12}>
                    <Typography variant="subtitle1">Last Run</Typography>
                    <Typography variant="body1">{projectStats.lastRun}</Typography>
                  </Grid>
                </Grid>
              ) : (
                <Typography>No data available</Typography>
              )}
            </CardContent>
            <CardActions>
              <Button 
                variant="outlined" 
                color="primary"
                onClick={() => navigate('/visualizations')}
                disabled={!projectStats}
              >
                View Visualizations
              </Button>
              <Button 
                variant="outlined"
                onClick={() => navigate('/results')}
                disabled={!projectStats}
              >
                View Results
              </Button>
            </CardActions>
          </Card>
        </Grid>

        {/* Error Metrics Card */}
        {projectStats && projectStats.errorMetrics && (
          <Grid item xs={12}>
            <Card>
              <CardHeader title="Kümeleme Kalitesi ve Hata Metrikleri" />
              <CardContent>
                <Grid container spacing={3}>
                  {/* Quality Overview */}
                  <Grid item xs={12} md={6}>
                    <Typography variant="h6" gutterBottom color="primary">
                      Genel Kalite Skoru
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                      <Typography variant="h3" sx={{ 
                        color: projectStats.errorMetrics.overall_quality_score > 70 ? 'green' : 
                               projectStats.errorMetrics.overall_quality_score > 50 ? 'orange' : 'red'
                      }}>
                        {Math.round(projectStats.errorMetrics.overall_quality_score)}
                      </Typography>
                      <Typography variant="h6" sx={{ ml: 1 }}>/100</Typography>
                    </Box>
                    <Typography variant="body2" color="text.secondary">
                      {projectStats.errorMetrics.is_high_quality ? 
                        '✅ Yüksek kaliteli kümeleme' : 
                        '⚠️ İyileştirme gerekiyor'}
                    </Typography>
                  </Grid>

                  {/* Balance Metrics */}
                  <Grid item xs={12} md={6}>
                    <Typography variant="h6" gutterBottom color="primary">
                      Küme Dengesi
                    </Typography>
                    <Typography variant="body1">
                      Denge Skoru: <strong>{(projectStats.errorMetrics.cluster_balance_score * 100).toFixed(1)}%</strong>
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Dengesizlik Oranı: {projectStats.errorMetrics.imbalance_ratio.toFixed(1)}x
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {projectStats.errorMetrics.imbalance_ratio < 3 ? 
                        '✅ Dengeli dağılım' : 
                        '⚠️ Dengesiz dağılım'}
                    </Typography>
                  </Grid>

                  {/* Error Statistics */}
                  <Grid item xs={12} md={4}>
                    <Typography variant="subtitle1" gutterBottom>
                      📊 Hata İstatistikleri
                    </Typography>
                    <Typography variant="body2">
                      Kategori Hatası: <strong>{projectStats.errorMetrics.category_error_rate.toFixed(1)}%</strong>
                    </Typography>
                    <Typography variant="body2">
                      Aykırı Değerler: <strong>{projectStats.errorMetrics.outlier_percentage.toFixed(1)}%</strong>
                    </Typography>
                    <Typography variant="body2">
                      Karışık Kümeler: <strong>{projectStats.errorMetrics.mixed_clusters}</strong>
                    </Typography>
                    <Typography variant="body2">
                      Saf Kümeler: <strong>{projectStats.errorMetrics.pure_clusters}</strong>
                    </Typography>
                  </Grid>

                  {/* Quality Metrics */}
                  <Grid item xs={12} md={4}>
                    <Typography variant="subtitle1" gutterBottom>
                      🎯 Kalite Metrikleri
                    </Typography>
                    <Typography variant="body2">
                      Silhouette Skoru: <strong>{projectStats.errorMetrics.silhouette_score.toFixed(3)}</strong>
                    </Typography>
                    <Typography variant="body2">
                      Ortalama Homojenlik: <strong>{projectStats.errorMetrics.avg_homogeneity.toFixed(1)}%</strong>
                    </Typography>
                    <Typography variant="body2">
                      Dağınık Kategoriler: <strong>{projectStats.errorMetrics.scattered_categories}</strong>
                    </Typography>
                  </Grid>

                  {/* Recommendations */}
                  <Grid item xs={12} md={4}>
                    <Typography variant="subtitle1" gutterBottom>
                      💡 Öneriler
                    </Typography>
                    {projectStats.errorMetrics.category_error_rate > 30 && (
                      <Typography variant="body2" color="error">
                        • Kategori filtrelerini gözden geçirin
                      </Typography>
                    )}
                    {projectStats.errorMetrics.imbalance_ratio > 5 && (
                      <Typography variant="body2" color="error">
                        • K değerini artırın
                      </Typography>
                    )}
                    {projectStats.errorMetrics.outlier_percentage > 15 && (
                      <Typography variant="body2" color="error">
                        • Veri temizleme yapın
                      </Typography>
                    )}
                    {projectStats.errorMetrics.overall_quality_score > 80 && (
                      <Typography variant="body2" color="success.main">
                        ✅ Kümeleme başarılı!
                      </Typography>
                    )}
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>
    </div>
  );
} 