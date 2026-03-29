import React, { useState, useMemo } from 'react';

/**
 * Professional DataTable Component
 * Features: Sort, Filter, Search, Pagination
 */
export function DataTable({ 
  data = [], 
  columns = [], 
  onRowClick = null,
  searchableFields = []
}) {
  const [sortConfig, setSortConfig] = useState({ key: null, order: 'asc' });
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 10;

  // Handle sorting
  const handleSort = (key) => {
    let order = 'asc';
    if (sortConfig.key === key && sortConfig.order === 'asc') {
      order = 'desc';
    }
    setSortConfig({ key, order });
  };

  // Filter & sort data
  const processedData = useMemo(() => {
    let result = [...data];

    // Search filter
    if (searchTerm) {
      result = result.filter(item =>
        searchableFields.some(field => {
          const value = String(item[field] || '').toLowerCase();
          return value.includes(searchTerm.toLowerCase());
        })
      );
    }

    // Sort
    if (sortConfig.key) {
      result.sort((a, b) => {
        const aVal = a[sortConfig.key];
        const bVal = b[sortConfig.key];

        if (aVal < bVal) return sortConfig.order === 'asc' ? -1 : 1;
        if (aVal > bVal) return sortConfig.order === 'asc' ? 1 : -1;
        return 0;
      });
    }

    return result;
  }, [data, sortConfig, searchTerm, searchableFields]);

  // Pagination
  const totalPages = Math.ceil(processedData.length / itemsPerPage);
  const startIdx = (currentPage - 1) * itemsPerPage;
  const paginatedData = processedData.slice(startIdx, startIdx + itemsPerPage);

  return (
    <div className="data-table-container">
      {/* Search Bar */}
      <div className="table-search-bar">
        <input
          type="text"
          placeholder="🔍 Search..."
          value={searchTerm}
          onChange={(e) => {
            setSearchTerm(e.target.value);
            setCurrentPage(1);
          }}
          className="search-input"
        />
        <span className="result-count">
          {processedData.length} of {data.length} records
        </span>
      </div>

      {/* Table */}
      <div className="table-wrapper">
        <table className="data-table">
          <thead>
            <tr>
              {columns.map((col) => (
                <th
                  key={col.key}
                  onClick={() => handleSort(col.key)}
                  className={`sortable ${sortConfig.key === col.key ? sortConfig.order : ''}`}
                >
                  {col.label}
                  {sortConfig.key === col.key && (
                    <span className="sort-icon">
                      {sortConfig.order === 'asc' ? ' ▲' : ' ▼'}
                    </span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {paginatedData.map((row, idx) => (
              <tr
                key={idx}
                onClick={() => onRowClick?.(row)}
                className={onRowClick ? 'clickable' : ''}
              >
                {columns.map((col) => (
                  <td key={col.key}>
                    {col.render ? col.render(row[col.key], row) : row[col.key]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      <div className="table-pagination">
        <button
          onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
          disabled={currentPage === 1}
        >
          ← Previous
        </button>
        <span>
          Page {currentPage} of {totalPages}
        </span>
        <button
          onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
          disabled={currentPage === totalPages}
        >
          Next →
        </button>
      </div>

      <style>{`
        .data-table-container {
          width: 100%;
          margin: 20px 0;
        }

        .table-search-bar {
          display: flex;
          gap: 10px;
          margin-bottom: 15px;
          align-items: center;
        }

        .search-input {
          flex: 1;
          padding: 10px;
          border: 1px solid #ddd;
          border-radius: 4px;
          font-size: 14px;
        }

        .result-count {
          color: #666;
          font-size: 13px;
        }

        .table-wrapper {
          overflow-x: auto;
          margin-bottom: 15px;
        }

        .data-table {
          width: 100%;
          border-collapse: collapse;
          background: white;
          box-shadow: 0 1px 3px rgba(0,0,0,0.1);
          border-radius: 4px;
          overflow: hidden;
        }

        .data-table thead {
          background: #f5f5f5;
          font-weight: 600;
        }

        .data-table th {
          padding: 12px;
          text-align: left;
          font-size: 13px;
          border-bottom: 2px solid #ddd;
        }

        .data-table th.sortable {
          cursor: pointer;
          user-select: none;
          transition: background 0.2s;
        }

        .data-table th.sortable:hover {
          background: #efefef;
        }

        .sort-icon {
          font-size: 11px;
          margin-left: 5px;
        }

        .data-table td {
          padding: 12px;
          border-bottom: 1px solid #eee;
          font-size: 13px;
        }

        .data-table tbody tr {
          transition: background 0.2s;
        }

        .data-table tbody tr.clickable:hover {
          background: #f9f9f9;
          cursor: pointer;
        }

        .table-pagination {
          display: flex;
          justify-content: center;
          gap: 15px;
          align-items: center;
          padding: 15px;
          background: #f5f5f5;
          border-radius: 4px;
        }

        .table-pagination button {
          padding: 8px 16px;
          border: 1px solid #ddd;
          background: white;
          border-radius: 4px;
          cursor: pointer;
          font-size: 13px;
          transition: all 0.2s;
        }

        .table-pagination button:hover:not(:disabled) {
          background: #007bff;
          color: white;
          border-color: #007bff;
        }

        .table-pagination button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
      `}</style>
    </div>
  );
}

export default DataTable;
