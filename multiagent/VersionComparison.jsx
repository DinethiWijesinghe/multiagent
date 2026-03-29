import React, { useState } from 'react';

/**
 * Version Comparison Component
 * Shows before/after view of data changes
 */
export function VersionComparison({
  versions = []
}) {
  const [selectedVersion, setSelectedVersion] = useState(null);
  const [compareMode, setCompareMode] = useState(false);
  const [compareWith, setCompareWith] = useState(null);

  if (!versions || versions.length === 0) {
    return (
      <div className="version-container">
        <p className="no-versions">No version history available</p>
      </div>
    );
  }

  const compareVersions = (v1, v2) => {
    const diff = {};
    const allKeys = new Set([
      ...Object.keys(v1.data || {}),
      ...Object.keys(v2.data || {})
    ]);

    allKeys.forEach(key => {
      const old = v1.data?.[key];
      const newVal = v2.data?.[key];
      
      if (JSON.stringify(old) !== JSON.stringify(newVal)) {
        diff[key] = { old, new: newVal };
      }
    });

    return diff;
  };

  const formatValue = (val) => {
    if (typeof val === 'object' && val !== null) {
      return JSON.stringify(val, null, 2);
    }
    return String(val || '');
  };

  const formatDate = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  const currentVersion = selectedVersion ? versions[selectedVersion] : versions[versions.length - 1];
  const comparisonDiff = compareMode && compareWith !== null
    ? compareVersions(versions[compareWith], currentVersion)
    : null;

  return (
    <div className="version-container">
      <h3>📜 Version History</h3>

      {/* Version Timeline */}
      <div className="version-timeline">
        {versions.map((version, idx) => (
          <div
            key={idx}
            className={`version-item ${selectedVersion === idx ? 'active' : ''}`}
            onClick={() => {
              setSelectedVersion(idx);
              setCompareMode(false);
            }}
          >
            <div className="version-badge">v{version.version}</div>
            <div className="version-info">
              <p className="version-action">{version.action}</p>
              <p className="version-time">{formatDate(version.timestamp)}</p>
              <p className="version-user">by {version.changed_by}</p>
            </div>
            {version.changes && (
              <div className="version-changes-indicator">
                {Object.keys(version.changes).length} changes
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Current Version Detail */}
      {currentVersion && (
        <div className="version-detail">
          <div className="detail-header">
            <h4>Version {currentVersion.version}: {currentVersion.action}</h4>
            <div className="detail-meta">
              <span className="meta-item">📅 {formatDate(currentVersion.timestamp)}</span>
              <span className="meta-item">👤 {currentVersion.changed_by}</span>
            </div>
          </div>

          {/* Comparison Controls */}
          {selectedVersion !== versions.length - 1 && (
            <div className="comparison-controls">
              <label>
                <input
                  type="checkbox"
                  checked={compareMode}
                  onChange={(e) => {
                    setCompareMode(e.target.checked);
                    if (e.target.checked && compareWith === null) {
                      setCompareWith(selectedVersion + 1);
                    }
                  }}
                />
                Compare with next version
              </label>
              {compareMode && (
                <select
                  value={compareWith || ''}
                  onChange={(e) => setCompareWith(Number(e.target.value))}
                >
                  {versions.slice(selectedVersion + 1).map((v, idx) => (
                    <option key={idx} value={selectedVersion + 1 + idx}>
                      v{v.version} ({formatDate(v.timestamp)})
                    </option>
                  ))}
                </select>
              )}
            </div>
          )}

          {/* Data Display */}
          {compareMode && comparisonDiff ? (
            <div className="version-comparison">
              <h5>Changes Made</h5>
              {Object.keys(comparisonDiff).length === 0 ? (
                <p className="no-changes">No changes in compared fields</p>
              ) : (
                Object.entries(comparisonDiff).map(([key, { old, new: newVal }]) => (
                  <div key={key} className="change-item">
                    <div className="change-field">{key}</div>
                    <div className="change-values">
                      <div className="old-value">
                        <strong>Before:</strong>
                        <pre>{formatValue(old)}</pre>
                      </div>
                      <div className="new-value">
                        <strong>After:</strong>
                        <pre>{formatValue(newVal)}</pre>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          ) : (
            <div className="version-data">
              <h5>Full Data</h5>
              <pre className="data-json">
                {JSON.stringify(currentVersion.data, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}

      <style>{`
        .version-container {
          background: white;
          border: 1px solid #dee2e6;
          border-radius: 6px;
          padding: 20px;
          margin: 20px 0;
        }

        .version-container h3 {
          margin: 0 0 15px 0;
          color: #333;
        }

        .no-versions {
          text-align: center;
          color: #999;
          padding: 30px;
        }

        .version-timeline {
          display: flex;
          gap: 10px;
          margin-bottom: 20px;
          overflow-x: auto;
          padding-bottom: 10px;
        }

        .version-item {
          flex-shrink: 0;
          background: #f8f9fa;
          border: 2px solid #dee2e6;
          border-radius: 6px;
          padding: 12px;
          cursor: pointer;
          transition: all 0.2s;
          min-width: 200px;
        }

        .version-item:hover {
          border-color: #007bff;
          background: #f0f7ff;
        }

        .version-item.active {
          border-color: #007bff;
          background: #007bff;
          color: white;
        }

        .version-badge {
          font-weight: 600;
          font-size: 12px;
          background: rgba(0, 0, 0, 0.1);
          padding: 4px 8px;
          border-radius: 3px;
          display: inline-block;
          margin-bottom: 8px;
        }

        .version-item.active .version-badge {
          background: rgba(255, 255, 255, 0.3);
        }

        .version-info p {
          margin: 4px 0;
          font-size: 12px;
        }

        .version-action {
          font-weight: 600;
        }

        .version-time {
          color: #666;
          font-size: 11px;
        }

        .version-item.active .version-time,
        .version-item.active .version-user {
          color: rgba(255, 255, 255, 0.8);
        }

        .version-user {
          color: #999;
          font-size: 11px;
          font-style: italic;
        }

        .version-changes-indicator {
          margin-top: 8px;
          font-size: 11px;
          color: #ffc107;
          font-weight: 600;
        }

        .version-detail {
          border-top: 1px solid #dee2e6;
          padding-top: 20px;
        }

        .detail-header {
          margin-bottom: 15px;
        }

        .detail-header h4 {
          margin: 0 0 8px 0;
          color: #333;
        }

        .detail-meta {
          display: flex;
          gap: 12px;
          font-size: 12px;
          color: #666;
        }

        .meta-item {
          display: flex;
          gap: 4px;
        }

        .comparison-controls {
          background: #f8f9fa;
          border: 1px solid #dee2e6;
          border-radius: 4px;
          padding: 12px;
          margin-bottom: 15px;
          display: flex;
          gap: 12px;
          align-items: center;
          flex-wrap: wrap;
        }

        .comparison-controls label {
          display: flex;
          align-items: center;
          gap: 6px;
          font-size: 13px;
          cursor: pointer;
        }

        .comparison-controls input[type="checkbox"] {
          cursor: pointer;
        }

        .comparison-controls select {
          padding: 6px 10px;
          border: 1px solid #ced4da;
          border-radius: 4px;
          font-size: 13px;
        }

        .version-comparison,
        .version-data {
          background: #f8f9fa;
          border: 1px solid #dee2e6;
          border-radius: 4px;
          padding: 15px;
        }

        .version-comparison h5,
        .version-data h5 {
          margin: 0 0 12px 0;
          color: #333;
          font-size: 13px;
        }

        .no-changes {
          color: #999;
          font-size: 13px;
          text-align: center;
          padding: 20px;
        }

        .change-item {
          margin-bottom: 15px;
          border-bottom: 1px solid #dee2e6;
          padding-bottom: 15px;
        }

        .change-item:last-child {
          border-bottom: none;
          margin-bottom: 0;
          padding-bottom: 0;
        }

        .change-field {
          font-weight: 600;
          color: #333;
          font-size: 13px;
          margin-bottom: 8px;
        }

        .change-values {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 12px;
        }

        .old-value,
        .new-value {
          display: flex;
          flex-direction: column;
          gap: 6px;
        }

        .old-value strong {
          color: #dc3545;
          font-size: 12px;
        }

        .new-value strong {
          color: #28a745;
          font-size: 12px;
        }

        .change-values pre {
          margin: 0;
          background: white;
          border: 1px solid #ced4da;
          padding: 8px;
          border-radius: 3px;
          font-size: 11px;
          max-height: 100px;
          overflow-y: auto;
          color: #333;
        }

        .data-json {
          background: white;
          border: 1px solid #ced4da;
          padding: 12px;
          border-radius: 4px;
          font-size: 11px;
          max-height: 300px;
          overflow-y: auto;
          color: #333;
          margin: 0;
        }

        @media (max-width: 768px) {
          .change-values {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </div>
  );
}

export default VersionComparison;
