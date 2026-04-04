import React, { useState } from 'react';

/**
 * Advanced Search & Filter Component
 * Multi-criteria filtering for tables
 */
export function SearchFilter({
  onFilterChange = () => {},
  filterOptions = {
    status: ['All', 'Incomplete', 'Complete', 'Approved', 'Rejected'],
    role: ['All', 'student', 'advisor', 'admin'],
    dateRange: true,
    customFilters: []
  }
}) {
  const [filters, setFilters] = useState({
    searchTerm: '',
    status: 'All',
    role: 'All',
    dateFrom: '',
    dateTo: '',
    customValues: {}
  });

  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleFilterChange = (field, value) => {
    const newFilters = { ...filters, [field]: value };
    setFilters(newFilters);
    onFilterChange(newFilters);
  };

  const handleReset = () => {
    const resetFilters = {
      searchTerm: '',
      status: 'All',
      role: 'All',
      dateFrom: '',
      dateTo: '',
      customValues: {}
    };
    setFilters(resetFilters);
    onFilterChange(resetFilters);
  };

  return (
    <div className="search-filter-container">
      {/* Quick Search */}
      <div className="quick-search">
        <input
          type="text"
          placeholder="🔍 yesQuick search by name, email..."
          value={filters.searchTerm}
          onChange={(e) => handleFilterChange('searchTerm', e.target.value)}
          className="search-input"
        />
      </div>

      {/* Filter Pills */}
      <div className="filter-pills">
        <label className="filter-pill">
          <span>Status:</span>
          <select
            value={filters.status}
            onChange={(e) => handleFilterChange('status', e.target.value)}
          >
            {filterOptions.status.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>

        <label className="filter-pill">
          <span>Role:</span>
          <select
            value={filters.role}
            onChange={(e) => handleFilterChange('role', e.target.value)}
          >
            {filterOptions.role.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
        </label>

        <button
          className="advanced-toggle"
          onClick={() => setShowAdvanced(!showAdvanced)}
        >
          {showAdvanced ? '− Advanced' : '+ Advanced'}
        </button>

        {(filters.searchTerm || filters.status !== 'All' || filters.role !== 'All') && (
          <button className="reset-btn" onClick={handleReset}>
            Clear Filters
          </button>
        )}
      </div>

      {/* Advanced Filters */}
      {showAdvanced && (
        <div className="advanced-filters">
          {filterOptions.dateRange && (
            <div className="filter-row">
              <label>Date Range:</label>
              <input
                type="date"
                value={filters.dateFrom}
                onChange={(e) => handleFilterChange('dateFrom', e.target.value)}
              />
              <span>to</span>
              <input
                type="date"
                value={filters.dateTo}
                onChange={(e) => handleFilterChange('dateTo', e.target.value)}
              />
            </div>
          )}

          {filterOptions.customFilters?.map((filter) => (
            <div key={filter.name} className="filter-row">
              <label>{filter.label}:</label>
              {filter.type === 'text' && (
                <input
                  type="text"
                  placeholder={filter.placeholder}
                  value={filters.customValues[filter.name] || ''}
                  onChange={(e) =>
                    handleFilterChange('customValues', {
                      ...filters.customValues,
                      [filter.name]: e.target.value
                    })
                  }
                />
              )}
              {filter.type === 'select' && (
                <select
                  value={filters.customValues[filter.name] || 'All'}
                  onChange={(e) =>
                    handleFilterChange('customValues', {
                      ...filters.customValues,
                      [filter.name]: e.target.value
                    })
                  }
                >
                  <option>All</option>
                  {filter.options.map((opt) => (
                    <option key={opt} value={opt}>
                      {opt}
                    </option>
                  ))}
                </select>
              )}
              {filter.type === 'range' && (
                <>
                  <input
                    type="number"
                    placeholder="Min"
                    min={filter.min}
                    max={filter.max}
                    value={filters.customValues[`${filter.name}_min`] || ''}
                    onChange={(e) =>
                      handleFilterChange('customValues', {
                        ...filters.customValues,
                        [`${filter.name}_min`]: e.target.value
                      })
                    }
                  />
                  <span>−</span>
                  <input
                    type="number"
                    placeholder="Max"
                    min={filter.min}
                    max={filter.max}
                    value={filters.customValues[`${filter.name}_max`] || ''}
                    onChange={(e) =>
                      handleFilterChange('customValues', {
                        ...filters.customValues,
                        [`${filter.name}_max`]: e.target.value
                      })
                    }
                  />
                </>
              )}
            </div>
          ))}
        </div>
      )}

      <style>{`
        .search-filter-container {
          background: white;
          border: 1px solid #dee2e6;
          border-radius: 6px;
          padding: 15px;
          margin-bottom: 20px;
        }

        .quick-search {
          margin-bottom: 12px;
        }

        .search-input {
          width: 100%;
          padding: 10px 12px;
          border: 1px solid #ced4da;
          border-radius: 4px;
          font-size: 14px;
          transition: border-color 0.2s;
        }

        .search-input:focus {
          outline: none;
          border-color: #007bff;
          box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.1);
        }

        .filter-pills {
          display: flex;
          gap: 10px;
          align-items: center;
          flex-wrap: wrap;
          margin-bottom: 12px;
        }

        .filter-pill {
          display: flex;
          gap: 6px;
          align-items: center;
          padding: 6px 12px;
          background: #f8f9fa;
          border: 1px solid #ced4da;
          border-radius: 20px;
          font-size: 13px;
        }

        .filter-pill span {
          font-weight: 600;
          color: #666;
        }

        .filter-pill select {
          border: none;
          background: transparent;
          font-size: 13px;
          cursor: pointer;
          padding: 0;
        }

        .advanced-toggle {
          padding: 6px 12px;
          background: #e9ecef;
          border: 1px solid #ced4da;
          border-radius: 20px;
          cursor: pointer;
          font-size: 13px;
          font-weight: 600;
          transition: all 0.2s;
        }

        .advanced-toggle:hover {
          background: #dee2e6;
        }

        .reset-btn {
          padding: 6px 12px;
          background: white;
          border: 1px solid #dc3545;
          border-radius: 20px;
          cursor: pointer;
          font-size: 13px;
          color: #dc3545;
          font-weight: 600;
          transition: all 0.2s;
        }

        .reset-btn:hover {
          background: #dc3545;
          color: white;
        }

        .advanced-filters {
          background: #f8f9fa;
          border-top: 1px solid #dee2e6;
          margin-top: 12px;
          padding: 12px;
          border-radius: 4px;
        }

        .filter-row {
          display: flex;
          gap: 10px;
          align-items: center;
          margin-bottom: 10px;
          flex-wrap: wrap;
        }

        .filter-row:last-child {
          margin-bottom: 0;
        }

        .filter-row label {
          font-weight: 600;
          font-size: 13px;
          color: #333;
          min-width: 80px;
        }

        .filter-row input,
        .filter-row select {
          padding: 6px 10px;
          border: 1px solid #ced4da;
          border-radius: 4px;
          font-size: 13px;
          flex: 1;
          min-width: 120px;
        }

        .filter-row span {
          color: #999;
          font-weight: 600;
        }
      `}</style>
    </div>
  );
}

export default SearchFilter;
