import React, { useState, useEffect } from 'react';
import {
  Typography, Paper, Grid, Card, CardContent, 
  CardHeader, CircularProgress, Box, Tabs, Tab,
  Button, Alert
} from '@mui/material';
import axios from 'axios';

export default function Visualizations() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [visualizations, setVisualizations] = useState(null);
  const [activeTab, setActiveTab] = useState(0);

  useEffect(() => {
    const fetchVisualizations = async () => {
      try {
        const response = await axios.get('/api/visualizations');
        setVisualizations(response.data);
        setLoading(false);
      } catch (err) {
        console.error('Error fetching visualizations:', err);
        setError('Failed to load visualizations. Server might not be running.');
        setLoading(false);
        
        // Fallback demo data
        setVisualizations({
          clusterSizesUrl: '/visualizations/cluster_sizes.html',
          categoryDistributionUrl: '/visualizations/category_distribution.png',
          clusterCategoryHeatmapUrl: '/visualizations/cluster_category_heatmap.png',
          wordcloudsUrl: '/visualizations/cluster_wordclouds.png'
        });
      }
    };

    fetchVisualizations();
  }, []);

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

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
              This interactive pie chart shows the distribution of papers across different clusters.
              The size of each slice represents the number of papers in that cluster.
            </Typography>
            
            <Box sx={{ p: 2, border: '1px solid #eee', borderRadius: 2, bgcolor: '#f9f9f9', mb: 2 }}>
              <iframe
                src={visualizations.clusterSizesUrl}
                style={{ width: '100%', height: '500px', border: 'none' }}
                title="Cluster Sizes"
              />
            </Box>
            
            <Typography variant="body2">
              <strong>Insight:</strong> Cluster 0 contains the majority of papers (747 papers, 98.4%), 
              suggesting that most of the papers are closely related in terms of their content.
            </Typography>
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