# Academic Paper Clustering Frontend

This is the React frontend for the Academic Paper Clustering project. It provides a modern web interface for collecting, clustering, and visualizing academic papers from ArXiv.

## Features

- Interactive dashboard with project statistics
- Data collection interface for ArXiv papers
- Configurable clustering parameters
- Visualization of clustering results
- Searchable and filterable papers list

## Getting Started

### Prerequisites

- Node.js (v14 or later)
- npm or yarn

### Installation

1. Install dependencies:

```bash
npm install
```

Or if you use yarn:

```bash
yarn install
```

### Development

To start the development server:

```bash
npm start
```

This will run the app in development mode. Open [http://localhost:3000](http://localhost:3000) to view it in the browser.

### Building for Production

To build the app for production:

```bash
npm run build
```

This builds the app for production to the `build` folder. It correctly bundles React in production mode and optimizes the build for the best performance.

## Connecting to Backend

The frontend is configured to connect to the Flask backend API running on `http://localhost:5000`. Make sure the backend server is running before using the frontend application.

## Project Structure

- `src/components/` - Reusable UI components
- `src/pages/` - Page components for different sections of the app
- `src/App.js` - Main application component with routing
- `src/index.js` - Entry point 