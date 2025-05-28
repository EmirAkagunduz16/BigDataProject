import React, { useState, useEffect } from 'react';
import {
  Typography, Paper, CircularProgress, Box, 
  Table, TableBody, TableCell, TableContainer, 
  TableHead, TableRow, TablePagination, Grid,
  Chip, TextField, InputAdornment, Card, CardContent,
  Alert, MenuItem, FormControl, InputLabel, Select
} from '@mui/material';
import { Search as SearchIcon } from '@mui/icons-material';
import axios from 'axios';

export default function Results() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [papers, setPapers] = useState([]);
  const [filteredPapers, setFilteredPapers] = useState([]);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterCluster, setFilterCluster] = useState('all');
  const [filterCategory, setFilterCategory] = useState('all');
  const [categories, setCategories] = useState([]);
  const [clusters, setClusters] = useState([]);

  useEffect(() => {
    const fetchResults = async () => {
      try {
        const response = await axios.get('/api/clustered-papers');
        setPapers(response.data.papers);
        setFilteredPapers(response.data.papers);
        
        // Extract unique categories and clusters
        const uniqueCategories = [...new Set(response.data.papers.map(paper => paper.primary_category))];
        const uniqueClusters = [...new Set(response.data.papers.map(paper => paper.cluster))];
        
        setCategories(uniqueCategories);
        setClusters(uniqueClusters);
        setLoading(false);
      } catch (err) {
        console.error('Error fetching papers:', err);
        setError('Failed to load papers. Server might not be running.');
        setLoading(false);
        
        // Fallback demo data
        const demoPapers = [
          {
            id: 'arxiv:2301.00001',
            title: 'How does Alignment Enhance LLMs Multilingual Capabilities? A Language Neurons Perspective',
            summary: 'This paper explores the multilingual capabilities of large language models...',
            authors: 'John Doe, Jane Smith',
            published: '2023-01-01',
            primary_category: 'cs.CL',
            cluster: 0
          },
          {
            id: 'arxiv:2301.00002',
            title: 'Improved Bounds for Swap Multicalibration and Swap Omniprediction',
            summary: 'We study the problem of swap multicalibration...',
            authors: 'Alice Johnson, Bob Brown',
            published: '2023-01-02',
            primary_category: 'cs.LG',
            cluster: 1
          },
          {
            id: 'arxiv:2301.00003',
            title: 'SageAttention2++: A More Efficient Implementation of SageAttention2',
            summary: 'We present an improved implementation of the SageAttention mechanism...',
            authors: 'Charlie Chen, Diana Davis',
            published: '2023-01-03',
            primary_category: 'cs.AI',
            cluster: 2
          },
          {
            id: 'arxiv:2301.00004',
            title: 'ViewSpatial-Bench: Evaluating Multi-perspective Spatial Localization in Vision-Language Models',
            summary: 'This paper introduces a benchmark for evaluating spatial understanding...',
            authors: 'Eva Evans, Frank Foster',
            published: '2023-01-04',
            primary_category: 'cs.CV',
            cluster: 0
          },
          {
            id: 'arxiv:2301.00005',
            title: 'Diffusion with stochastic resetting on a lattice',
            summary: 'We study diffusion processes with stochastic resetting on discrete lattices...',
            authors: 'Grace Green, Henry Hill',
            published: '2023-01-05',
            primary_category: 'cond-mat.stat-mech',
            cluster: 1
          }
        ];
        
        setPapers(demoPapers);
        setFilteredPapers(demoPapers);
        
        // Extract unique categories and clusters from demo data
        const uniqueCategories = [...new Set(demoPapers.map(paper => paper.primary_category))];
        const uniqueClusters = [...new Set(demoPapers.map(paper => paper.cluster))];
        
        setCategories(uniqueCategories);
        setClusters(uniqueClusters);
      }
    };

    fetchResults();
  }, []);

  useEffect(() => {
    // Filter papers based on search term and filters
    let filtered = papers;
    
    if (searchTerm) {
      const lowercasedSearch = searchTerm.toLowerCase();
      filtered = filtered.filter(paper => 
        paper.title.toLowerCase().includes(lowercasedSearch) || 
        paper.summary.toLowerCase().includes(lowercasedSearch) ||
        paper.authors.toLowerCase().includes(lowercasedSearch)
      );
    }
    
    if (filterCluster !== 'all') {
      filtered = filtered.filter(paper => paper.cluster === parseInt(filterCluster));
    }
    
    if (filterCategory !== 'all') {
      filtered = filtered.filter(paper => paper.primary_category === filterCategory);
    }
    
    setFilteredPapers(filtered);
    setPage(0); // Reset to first page when filters change
  }, [searchTerm, filterCluster, filterCategory, papers]);

  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const handleSearchChange = (event) => {
    setSearchTerm(event.target.value);
  };

  const handleClusterFilterChange = (event) => {
    setFilterCluster(event.target.value);
  };

  const handleCategoryFilterChange = (event) => {
    setFilterCategory(event.target.value);
  };

  const getClusterColor = (cluster) => {
    const colors = ['primary', 'secondary', 'error', 'warning', 'info', 'success'];
    return colors[cluster % colors.length];
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
        Clustering Results
      </Typography>
      
      {error && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}
      
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Filter Results
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Search Papers"
                  variant="outlined"
                  value={searchTerm}
                  onChange={handleSearchChange}
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        <SearchIcon />
                      </InputAdornment>
                    ),
                  }}
                  placeholder="Search by title, summary, or author"
                />
              </Grid>
              <Grid item xs={6} md={3}>
                <FormControl fullWidth>
                  <InputLabel>Cluster</InputLabel>
                  <Select
                    value={filterCluster}
                    label="Cluster"
                    onChange={handleClusterFilterChange}
                  >
                    <MenuItem value="all">All Clusters</MenuItem>
                    {clusters.map(cluster => (
                      <MenuItem key={cluster} value={cluster}>
                        Cluster {cluster}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={6} md={3}>
                <FormControl fullWidth>
                  <InputLabel>Category</InputLabel>
                  <Select
                    value={filterCategory}
                    label="Category"
                    onChange={handleCategoryFilterChange}
                  >
                    <MenuItem value="all">All Categories</MenuItem>
                    {categories.map(category => (
                      <MenuItem key={category} value={category}>
                        {category}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
            </Grid>
          </Box>
          
          <Typography variant="body2" color="text.secondary">
            Showing {filteredPapers.length} of {papers.length} papers
          </Typography>
        </CardContent>
      </Card>
      
      <Paper>
        <TableContainer>
          <Table sx={{ minWidth: 650 }}>
            <TableHead>
              <TableRow>
                <TableCell>Title</TableCell>
                <TableCell>Authors</TableCell>
                <TableCell>Category</TableCell>
                <TableCell>Cluster</TableCell>
                <TableCell>Published</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {filteredPapers
                .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                .map((paper) => (
                  <TableRow key={paper.id} hover>
                    <TableCell>
                      <Typography variant="subtitle2">
                        {paper.title}
                      </Typography>
                      <Typography variant="body2" color="text.secondary" noWrap>
                        {paper.summary.substring(0, 120)}...
                      </Typography>
                    </TableCell>
                    <TableCell>{paper.authors.split(',')[0]}{paper.authors.split(',').length > 1 ? ' et al.' : ''}</TableCell>
                    <TableCell>
                      <Chip label={paper.primary_category} size="small" />
                    </TableCell>
                    <TableCell>
                      <Chip 
                        label={`Cluster ${paper.cluster}`} 
                        size="small"
                        color={getClusterColor(paper.cluster)}
                      />
                    </TableCell>
                    <TableCell>{new Date(paper.published).toLocaleDateString()}</TableCell>
                  </TableRow>
                ))}
              {filteredPapers.length === 0 && (
                <TableRow>
                  <TableCell colSpan={5} align="center">
                    No papers found matching your filters
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </TableContainer>
        <TablePagination
          rowsPerPageOptions={[5, 10, 25, 50]}
          component="div"
          count={filteredPapers.length}
          rowsPerPage={rowsPerPage}
          page={page}
          onPageChange={handleChangePage}
          onRowsPerPageChange={handleChangeRowsPerPage}
        />
      </Paper>
    </div>
  );
} 