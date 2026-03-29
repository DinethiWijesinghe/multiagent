import React from 'react';

/**
 * Data Validation Badge Component
 * Shows data quality: Valid, Incomplete, Error
 */
export function DataValidationBadge({
  data = {},
  requiredFields = [],
  validationRules = {}
}) {
  // Validate data
  const validate = () => {
    const issues = [];
    const completed = [];

    requiredFields.forEach(field => {
      const value = data[field];
      const isEmpty = !value || (typeof value === 'object' && Object.keys(value).length === 0);

      if (isEmpty) {
        issues.push(`${field} is required`);
      } else {
        completed.push(field);
      }
    });

    // Check validation rules (if provided)
    Object.entries(validationRules).forEach(([field, rule]) => {
      const value = data[field];

      if (!value && rule.required) {
        issues.push(`${field} is required`);
        return;
      }

      if (value) {
        // Type checking
        if (rule.type === 'email' && !/.+@.+\..+/.test(value)) {
          issues.push(`${field} must be a valid email`);
        }

        // Min/Max length
        if (rule.minLength && value.length < rule.minLength) {
          issues.push(`${field} must be at least ${rule.minLength} characters`);
        }
        if (rule.maxLength && value.length > rule.maxLength) {
          issues.push(`${field} must be no more than ${rule.maxLength} characters`);
        }

        // Min/Max value
        if (rule.min !== undefined && Number(value) < rule.min) {
          issues.push(`${field} must be at least ${rule.min}`);
        }
        if (rule.max !== undefined && Number(value) > rule.max) {
          issues.push(`${field} must be no more than ${rule.max}`);
        }

        if (!issues.find(i => i.includes(field))) {
          completed.push(field);
        }
      }
    });

    const completionScore = Math.round((completed.length / requiredFields.length) * 100);

    return {
      isValid: issues.length === 0,
      isComplete: completed.length === requiredFields.length,
      issues,
      completionScore,
      completed: completed.length,
      total: requiredFields.length
    };
  };

  const validation = validate();

  const getStatus = () => {
    if (validation.isValid) return { label: '✅ Valid', color: '#28a745', icon: '✓' };
    if (validation.completionScore >= 75) return { label: '⚠️ Incomplete', color: '#ffc107', icon: '!' };
    return { label: '❌ Error', color: '#dc3545', icon: '✕' };
  };

  const status = getStatus();

  return (
    <div className="validation-badge">
      <div className="badge-header">
        <div
          className="badge-status"
          style={{ backgroundColor: status.color }}
          title={validation.issues.join(', ')}
        >
          {status.label}
        </div>
        <div className="badge-score">
          {validation.completionScore}% Complete
          <span className="score-fraction"> ({validation.completed}/{validation.total})</span>
        </div>
      </div>

      {validation.issues.length > 0 && (
        <div className="badge-issues">
          <strong>Issues:</strong>
          <ul>
            {validation.issues.map((issue, idx) => (
              <li key={idx}>• {issue}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="badge-progress">
        <div className="progress-bar">
          <div
            className="progress-fill"
            style={{
              width: `${validation.completionScore}%`,
              backgroundColor: status.color
            }}
          />
        </div>
      </div>

      <style>{`
        .validation-badge {
          background: #f8f9fa;
          border: 1px solid #dee2e6;
          border-radius: 6px;
          padding: 12px;
          margin: 10px 0;
        }

        .badge-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 10px;
        }

        .badge-status {
          padding: 6px 12px;
          border-radius: 4px;
          color: white;
          font-weight: 600;
          font-size: 13px;
          display: inline-block;
        }

        .badge-score {
          font-size: 13px;
          font-weight: 600;
          color: #333;
        }

        .score-fraction {
          font-size: 12px;
          color: #666;
          font-weight: normal;
        }

        .badge-issues {
          background: white;
          border: 1px solid #ffeaea;
          border-radius: 4px;
          padding: 8px;
          margin: 10px 0;
          font-size: 12px;
        }

        .badge-issues strong {
          color: #dc3545;
          display: block;
          margin-bottom: 5px;
        }

        .badge-issues ul {
          list-style: none;
          padding: 0;
          margin: 0;
        }

        .badge-issues li {
          color: #666;
          margin: 4px 0;
          font-size: 12px;
        }

        .badge-progress {
          margin-top: 10px;
        }

        .progress-bar {
          width: 100%;
          height: 6px;
          background: #e9ecef;
          border-radius: 3px;
          overflow: hidden;
        }

        .progress-fill {
          height: 100%;
          transition: width 0.3s ease;
          border-radius: 3px;
        }
      `}</style>
    </div>
  );
}

export default DataValidationBadge;
