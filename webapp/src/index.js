window.CESIUM_BASE_URL = "/static";

/* eslint-disable import/first */
import React from 'react';
import ReactDOM from 'react-dom/client';
import "cesium/Build/Cesium/Widgets/widgets.css";
import App from './App';
import './App.css';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
