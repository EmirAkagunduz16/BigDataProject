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
          totalPapers: 759,
          totalClusters: 3,
          categoriesCount: 15,
          clusterSizes: [747, 7, 5],
          processingStatus: 'completed',
          lastRun: '2023-05-28 11:54:33'
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
      </Grid>
    </div>
  );
} 