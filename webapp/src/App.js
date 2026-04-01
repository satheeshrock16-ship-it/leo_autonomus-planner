import React, { useState, useEffect } from 'react';
import './App.css';
import EarthViewer from './components/EarthViewer';

function App() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [selectedObject, setSelectedObject] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);
        const res = await fetch("http://127.0.0.1:8000/objects");
        const json = await res.json();
        console.log("API DATA:", json);
        setData(json);
        setLastUpdate(new Date().toLocaleTimeString());
      } catch (err) {
        setError(err.message);
        console.error("FETCH ERROR:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <h1>LEO Autonomous Planner</h1>
        <div className="header-status">
          {lastUpdate && <span>Last updated: {lastUpdate}</span>}
        </div>
      </header>

      {/* Main Layout */}
      <div className="main-layout">
        {/* Left Sidebar - Details */}
        <aside className="sidebar left-sidebar">
          <div className="panel">
            <h2>Orbital Data</h2>
            {loading && <p className="loading">Loading...</p>}
            {error && <p className="error">Error: {error}</p>}
            {data && data.analysis ? (
              <div className="data-display">
                <div className="data-item">
                  <label>Collision Probability:</label>
                  <span className="value">{(data.analysis.collision_probability * 100).toFixed(2)}%</span>
                </div>
                <div className="data-item">
                  <label>Miss Distance:</label>
                  <span className="value">{data.analysis.miss_distance_km.toFixed(2)} km</span>
                </div>
                <div className="data-item">
                  <label>Delta-V Required:</label>
                  <span className="value">{data.analysis.delta_v_km_s.toFixed(4)} km/s</span>
                </div>
                <div className="data-item">
                  <label>TCA Time:</label>
                  <span className="value">{data.analysis.tca_time}</span>
                </div>
                <div className="data-item">
                  <label>Runtime:</label>
                  <span className="value">{data.analysis.runtime_seconds.toFixed(2)}s</span>
                </div>
              </div>
            ) : (
              <p className="no-data">No data available - waiting for first analysis...</p>
            )}
          </div>
        </aside>

        {/* Center - Main View */}
        <main className="main-content">
          <div className="panel earth-view">
            <h2>Earth View</h2>
            <EarthViewer
              objects={data?.objects}
              collisions={data?.collisions}
              onObjectClick={setSelectedObject}
            />
          </div>
        </main>

        {/* Right Sidebar - Status */}
        <aside className="sidebar right-sidebar">
          <div className="panel">
            <h2>System Status</h2>
            {data && data.analysis ? (
              <div className="status-display">
                <div className="status-card">
                  <h3>Highest Risk Encounter</h3>
                  <p className="status-value">{(data.analysis.miss_distance_km < 5 ? 'CRITICAL' : 'ELEVATED').toString()}</p>
                </div>
                <div className="status-card">
                  <h3>Recommended Action</h3>
                  <p className="status-value">{data.analysis.delta_v_km_s > 0 ? 'EXECUTE MANEUVER' : 'MONITOR'}</p>
                </div>
                <div className="status-card">
                  <h3>Analysis Status</h3>
                  <p className="status-value">COMPLETE</p>
                </div>
              </div>
            ) : (
              <p className="no-data">Awaiting analysis...</p>
            )}
            {selectedObject && (
              <div style={{ marginTop: "10px", padding: "10px", border: "1px solid cyan" }}>
                <h3>Selected Object</h3>
                <p>ID: {selectedObject.id}</p>
                <p>Type: {selectedObject.type}</p>
                <p>Altitude: {selectedObject.altitude} km</p>
                <p>Velocity: {selectedObject.velocity} km/s</p>
              </div>
            )}
          </div>
        </aside>
      </div>

      {/* Bottom Panel - Collision Alerts */}
      <footer className="footer">
        <div className="panel">
          <h2>Collision Alerts</h2>
          {data && data.analysis ? (
            <div className="alerts-display">
              <div className={`alert alert-${data.analysis.collision_probability > 0.01 ? 'critical' : 'info'}`}>
                <strong>Conjunction Alert:</strong> Miss distance {data.analysis.miss_distance_km.toFixed(2)} km, 
                Probability {(data.analysis.collision_probability * 100).toFixed(2)}%
              </div>
            </div>
          ) : (
            <p className="no-alerts">No active alerts</p>
          )}
        </div>
      </footer>
    </div>
  );
}

export default App;
