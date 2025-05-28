import React, { useState, useEffect } from 'react';
import {
  Typography, Paper, Card, CardContent, 
  CardHeader, CircularProgress, Box, Tabs, Tab,
  Button, Alert
} from '@mui/material';
import axios from 'axios';

export default function Visualizations() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [visualizations, setVisualizations] = useState(null);
  const [activeTab, setActiveTab] = useState(0);
  const [iframeError, setIframeError] = useState(false);
  const [projectStats, setProjectStats] = useState(null); // Gerçek veri için

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Visualizations ve stats'ı aynı anda çek
        const [visualizationsResponse, statsResponse] = await Promise.all([
          axios.get('/api/visualizations'),
          axios.get('/api/stats')
        ]);
        
        setVisualizations(visualizationsResponse.data);
        setProjectStats(statsResponse.data);
        
        // Check if cluster sizes file exists, if not set iframe error
        if (!visualizationsResponse.data.filesExist?.clusterSizesUrl) {
          setIframeError(true);
        }
        
        setLoading(false);
      } catch (err) {
        console.error('Error fetching data:', err);
        setError('Failed to load visualizations. Server might not be running.');
        setLoading(false);
        setIframeError(true);
        
        // Fallback demo data
        setVisualizations({
          clusterSizesUrl: '/visualizations/cluster_sizes.html',
          categoryDistributionUrl: '/visualizations/category_distribution.png',
          clusterCategoryHeatmapUrl: '/visualizations/cluster_category_heatmap.png',
          wordcloudsUrl: '/visualizations/cluster_wordclouds.png',
          filesExist: {
            clusterSizesUrl: false,
            categoryDistributionUrl: false,
            clusterCategoryHeatmapUrl: false,
            wordcloudsUrl: false
          }
        });
        
        // Fallback stats
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
          }
        });
      }
    };

    fetchData();
  }, []);

  // Cluster verilerini hesapla
  const getClusterData = () => {
    if (!projectStats || !projectStats.clusterSizes) {
      return [];
    }

    const total = projectStats.totalPapers;
    const clusterEntries = Object.entries(projectStats.clusterSizes)
      .map(([clusterId, size]) => ({
        id: clusterId,
        size: size,
        percentage: ((size / total) * 100).toFixed(1)
      }))
      .sort((a, b) => b.size - a.size); // Büyükten küçüğe sırala

    return clusterEntries;
  };

  // Renk paleti
  const getClusterColor = (index) => {
    const colors = ['#1976d2', '#dc004e', '#ed6c02', '#2e7d32', '#9c27b0', '#ff6f00', '#00796b', '#5d4037'];
    return colors[index % colors.length];
  };

  // CSS conic-gradient oluştur
  const generateConicGradient = (clusterData) => {
    let currentPercentage = 0;
    const gradientStops = clusterData.map((cluster, index) => {
      const startPercentage = currentPercentage;
      currentPercentage += parseFloat(cluster.percentage);
      const endPercentage = currentPercentage;
      const color = getClusterColor(index);
      
      return `${color} ${startPercentage}% ${endPercentage}%`;
    });
    
    return `conic-gradient(${gradientStops.join(', ')})`;
  };

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const handleIframeError = () => {
    setIframeError(true);
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="80vh">
        <CircularProgress />
      </Box>
    );
  }

  const clusterData = getClusterData();

  return (
    <div>
      <Typography variant="h4" gutterBottom>
        Clustering Visualizations
      </Typography>
      
      {error && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}
      
      <Paper sx={{ mb: 3 }}>
        <Tabs
          value={activeTab}
          onChange={handleTabChange}
          indicatorColor="primary"
          textColor="primary"
          variant="fullWidth"
        >
          <Tab label="Cluster Sizes" />
          <Tab label="Category Distribution" />
          <Tab label="Cluster-Category Heatmap" />
          <Tab label="Word Clouds" />
        </Tabs>
      </Paper>
      
      {activeTab === 0 && visualizations && (
        <Card>
          <CardHeader title="Cluster Size Distribution" />
          <CardContent>
            <Typography variant="body2" color="text.secondary" paragraph>
              Bu pasta grafiği, farklı kümelerdeki makalelerin dağılımını gösterir.
              Her dilimin boyutu, o kümedeki makale sayısını temsil eder.
            </Typography>
            
            {!iframeError ? (
              <Box sx={{ p: 2, border: '1px solid #eee', borderRadius: 2, bgcolor: '#f9f9f9', mb: 2 }}>
                <iframe
                  src={visualizations.clusterSizesUrl}
                  style={{ width: '100%', height: '500px', border: 'none' }}
                  title="Cluster Sizes"
                  onError={handleIframeError}
                  onLoad={(e) => {
                    // Iframe yüklendi, hata olmadığını kontrol et
                    try {
                      if (e.target.contentDocument === null) {
                        handleIframeError();
                      }
                    } catch (error) {
                      handleIframeError();
                    }
                  }}
                />
              </Box>
            ) : (
              <Box sx={{ p: 4, border: '1px solid #eee', borderRadius: 2, bgcolor: '#f9f9f9', mb: 2, textAlign: 'center' }}>
                <Alert severity="info" sx={{ mb: 2 }}>
                  İnteraktif görselleştirme yüklenemedi. Demo veri ile gösterim:
                </Alert>
                
                {/* Demo pie chart using CSS */}
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', mb: 3 }}>
                  <Box sx={{ position: 'relative', width: 300, height: 300 }}>
                    {/* Pie chart simulation with CSS */}
                    <Box
                      sx={{
                        width: 300,
                        height: 300,
                        borderRadius: '50%',
                        background: generateConicGradient(clusterData),
                        position: 'relative',
                        boxShadow: '0 4px 8px rgba(0,0,0,0.1)'
                      }}
                    />
                    
                    {/* Center label */}
                    <Box
                      sx={{
                        position: 'absolute',
                        top: '50%',
                        left: '50%',
                        transform: 'translate(-50%, -50%)',
                        backgroundColor: 'white',
                        borderRadius: '50%',
                        width: 120,
                        height: 120,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        flexDirection: 'column',
                        boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                      }}
                    >
                      <Typography variant="h6" color="primary">
                        {projectStats?.totalPapers || 639}
                      </Typography>
                      <Typography variant="caption">Toplam</Typography>
                    </Box>
                  </Box>
                </Box>
                
                {/* Legend */}
                <Box sx={{ display: 'flex', justifyContent: 'center', flexWrap: 'wrap', gap: 2 }}>
                  {clusterData.map((cluster, index) => (
                    <Box key={cluster.id} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Box sx={{ width: 16, height: 16, backgroundColor: getClusterColor(index) }} />
                      <Typography variant="body2">
                        Küme {parseInt(cluster.id) + 1} ({cluster.percentage}% - {cluster.size} makale)
                      </Typography>
                    </Box>
                  ))}
                </Box>
              </Box>
            )}
            
            <Typography variant="body2">
              <strong>Analiz:</strong> En büyük küme olan "Yapay Zeka ve Dil Modelleri" makalelerin 
              çoğunluğunu içermekte (%65). Bu, veri setindeki makalelerin büyük bir kısmının 
              modern AI araştırmaları ile ilgili olduğunu göstermektedir.
            </Typography>
            
            <Box mt={2}>
              <Button 
                variant="outlined" 
                onClick={() => {
                  if (visualizations?.clusterSizesUrl && visualizations?.filesExist?.clusterSizesUrl) {
                    window.open(visualizations.clusterSizesUrl, '_blank');
                  }
                }}
                disabled={!visualizations?.filesExist?.clusterSizesUrl}
              >
                Tam Ekranda Aç
              </Button>
              
              {iframeError && (
                <Typography variant="caption" color="text.secondary" sx={{ ml: 2 }}>
                  Görselleştirme dosyası bulunamadı. Lütfen önce kümeleme analizi çalıştırın.
                </Typography>
              )}
            </Box>
          </CardContent>
        </Card>
      )}
      
      {activeTab === 1 && visualizations && (
        <Card>
          <CardHeader title="Category Distribution" />
          <CardContent>
            <Typography variant="body2" color="text.secondary" paragraph>
              This bar chart shows the distribution of papers across different ArXiv categories.
              It helps identify which research areas are most represented in the dataset.
            </Typography>
            
            <Box 
              component="img"
              src={visualizations.categoryDistributionUrl}
              alt="Category Distribution"
              sx={{ 
                width: '100%',
                maxHeight: '600px',
                objectFit: 'contain',
                border: '1px solid #eee',
                borderRadius: 2,
                mb: 2
              }}
            />
            
            <Typography variant="body2">
              <strong>Insight:</strong> The most common categories are cs.LG (Machine Learning), 
              cs.CV (Computer Vision), and cs.CL (Computational Linguistics), reflecting the 
              current trends in AI research.
            </Typography>
          </CardContent>
        </Card>
      )}
      
      {activeTab === 2 && visualizations && (
        <Card>
          <CardHeader title="Cluster-Category Relationship" />
          <CardContent>
            <Typography variant="body2" color="text.secondary" paragraph>
              This heatmap shows the relationship between clusters and ArXiv categories.
              Each cell represents the number of papers that belong to both a specific cluster 
              and a specific category.
            </Typography>
            
            <Box 
              component="img"
              src={visualizations.clusterCategoryHeatmapUrl}
              alt="Cluster-Category Heatmap"
              sx={{ 
                width: '100%',
                maxHeight: '600px',
                objectFit: 'contain',
                border: '1px solid #eee',
                borderRadius: 2,
                mb: 2
              }}
            />
            
            <Typography variant="body2">
              <strong>Insight:</strong> Cluster 0 contains papers from all categories, while 
              Clusters 1 and 2 are more specialized. This suggests that the clustering algorithm 
              has identified both general and niche research areas.
            </Typography>
          </CardContent>
        </Card>
      )}
      
      {activeTab === 3 && visualizations && (
        <Card>
          <CardHeader title="Cluster Word Clouds" />
          <CardContent>
            <Typography variant="body2" color="text.secondary" paragraph>
              These word clouds show the most frequent terms in each cluster.
              The size of each word represents its frequency and importance within the cluster.
            </Typography>
            
            <Box 
              component="img"
              src={visualizations.wordcloudsUrl}
              alt="Cluster Word Clouds"
              sx={{ 
                width: '100%',
                maxHeight: '600px',
                objectFit: 'contain',
                border: '1px solid #eee',
                borderRadius: 2,
                mb: 2
              }}
            />
            
            <Typography variant="body2">
              <strong>Insight:</strong> Cluster 0 contains general AI terms like "models," "learning," 
              and "framework." Cluster 1 focuses on geometry and probability, while Cluster 2 focuses 
              on statistical methods and experimental designs.
            </Typography>
          </CardContent>
        </Card>
      )}
      
      <Box display="flex" justifyContent="center" mt={3}>
        <Button 
          variant="outlined" 
          color="primary"
          href={visualizations?.clusterSizesUrl}
          target="_blank"
          size="large"
          sx={{ mr: 2 }}
        >
          Open Interactive Chart
        </Button>
        <Button 
          variant="outlined"
          href="/api/download-visualizations"
          size="large"
        >
          Download All Visualizations
        </Button>
      </Box>
    </div>
  );
} 