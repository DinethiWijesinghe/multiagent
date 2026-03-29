import React, { useState } from 'react';

/**
 * Audit Log Component
 * Displays timeline of actions: who accessed what, when, and what changed
 */
export function AuditLog({ logs = [] }) {
  const [expandedLog, setExpandedLog] = useState(null);

  const getActionColor = (action) => {
    const colors = {
      'VIEW': '#17a2b8',      // blue
      'CREATE': '#28a745',    // green
      'UPDATE': '#ffc107',    // yellow
      'DELETE': '#dc3545',    // red
      'LOGIN': '#007bff',     // dark blue
      'LOGOUT': '#6c757d',    // gray
      'EXPORT': '#6f42c1'     // purple
    };
    return colors[action] || '#495057';
  };

  const getActionIcon = (action) => {
    const icons = {
      'VIEW': '👁️ ',
      'CREATE': '➕ ',
      'UPDATE': '✏️ ',
      'DELETE': '🗑️ ',
      'LOGIN': '🔓 ',
      'LOGOUT': '🔒 ',
      'EXPORT': '📥 '
    };
    return icons[action] || '• ';
  };

  const formatTime = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  return (
    <div className="audit-log-container">
      <h3>📋 Audit Log</h3>
      <div className="audit-timeline">
        {logs.length === 0 ? (
          <p className="no-logs">No audit logs available</p>
        ) : (
          logs.map((log, idx) => (
            <div key={idx} className="audit-entry">
              <div
                className="audit-marker"
                style={{ backgroundColor: getActionColor(log.action) }}
                onClick={() => setExpandedLog(expandedLog === idx ? null : idx)}
              >
                {getActionIcon(log.action)}
              </div>

              <div className="audit-content">
                <div
                  className="audit-header"
                  onClick={() => setExpandedLog(expandedLog === idx ? null : idx)}
                  style={{ cursor: 'pointer' }}
                >
                  <span className="action-badge" style={{ backgroundColor: getActionColor(log.action) }}>
                    {log.action}
                  </span>
                  <span className="actor">{log.actor_email || log.actor || 'System'}</span>
                  <span className="role-badge">{log.actor_role || 'user'}</span>
                  <span className="time">{formatTime(log.timestamp)}</span>
                </div>

                <div className="audit-details">
                  <p><strong>Action:</strong> {log.details || 'No details'}</p>
                  {log.target_email && (
                    <p><strong>Target:</strong> {log.target_email}</p>
                  )}
                  {log.ip_address && (
                    <p><strong>IP Address:</strong> {log.ip_address}</p>
                  )}
                </div>

                {expandedLog === idx && log.changes && (
                  <div className="audit-changes">
                    <strong>Changes Made:</strong>
                    <pre>{JSON.stringify(log.changes, null, 2)}</pre>
                  </div>
                )}
              </div>
            </div>
          ))
        )}
      </div>

      <style>{`
        .audit-log-container {
          background: white;
          border: 1px solid #dee2e6;
          border-radius: 6px;
          padding: 20px;
          margin: 20px 0;
        }

        .audit-log-container h3 {
          margin: 0 0 15px 0;
          color: #333;
        }

        .no-logs {
          text-align: center;
          color: #999;
          padding: 30px;
          font-size: 14px;
        }

        .audit-timeline {
          position: relative;
          padding-left: 30px;
        }

        .audit-timeline::before {
          content: '';
          position: absolute;
          left: 6px;
          top: 0;
          bottom: 0;
          width: 2px;
          background: #dee2e6;
        }

        .audit-entry {
          display: flex;
          margin-bottom: 20px;
          position: relative;
        }

        .audit-marker {
          width: 20px;
          height: 20px;
          border-radius: 50%;
          background: #007bff;
          position: absolute;
          left: -36px;
          top: 2px;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 10px;
          cursor: pointer;
          transition: transform 0.2s;
        }

        .audit-marker:hover {
          transform: scale(1.15);
        }

        .audit-content {
          flex: 1;
          background: #f8f9fa;
          border: 1px solid #dee2e6;
          border-radius: 4px;
          padding: 12px;
          transition: all 0.2s;
        }

        .audit-content:hover {
          background: #fff;
          border-color: #999;
        }

        .audit-header {
          display: flex;
          gap: 10px;
          align-items: center;
          flex-wrap: wrap;
          margin-bottom: 8px;
          user-select: none;
        }

        .action-badge {
          padding: 4px 8px;
          border-radius: 3px;
          color: white;
          font-weight: 600;
          font-size: 11px;
        }

        .actor {
          font-weight: 600;
          color: #333;
          font-size: 13px;
        }

        .role-badge {
          background: #e9ecef;
          padding: 2px 6px;
          border-radius: 3px;
          font-size: 11px;
          color: #666;
          font-style: italic;
        }

        .time {
          margin-left: auto;
          color: #999;
          font-size: 12px;
        }

        .audit-details {
          font-size: 12px;
          color: #666;
          line-height: 1.6;
        }

        .audit-details p {
          margin: 4px 0;
        }

        .audit-details strong {
          color: #333;
          min-width: 60px;
        }

        .audit-changes {
          background: white;
          border: 1px solid #e9ecef;
          border-radius: 3px;
          padding: 10px;
          margin-top: 10px;
          font-size: 11px;
        }

        .audit-changes strong {
          display: block;
          margin-bottom: 8px;
          color: #333;
        }

        .audit-changes pre {
          margin: 0;
          background: #f5f5f5;
          padding: 8px;
          border-radius: 3px;
          overflow-x: auto;
          max-height: 150px;
          font-size: 10px;
          color: #333;
        }
      `}</style>
    </div>
  );
}

export default AuditLog;
