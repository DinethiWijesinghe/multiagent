# 📊 Professional Data Viewing Components - Usage Guide

## Overview

Five production-ready React components for building professional data viewing dashboards:

1. **DataTable** - Sortable, filterable, paginated tables
2. **DataValidationBadge** - Show data quality/completeness
3. **AuditLog** - Timeline of all access and changes
4. **SearchFilter** - Advanced multi-criteria filtering
5. **VersionComparison** - Before/after data changes

---

## 🎯 Quick Start

### Import Components

```jsx
import DataTable from './DataTable';
import DataValidationBadge from './DataValidationBadge';
import AuditLog from './AuditLog';
import SearchFilter from './SearchFilter';
import VersionComparison from './VersionComparison';
```

---

## 📋 Component Reference

### 1. DataTable Component

**Purpose:** Display tabular data with sorting, searching, filtering, and pagination.

**Props:**
```jsx
<DataTable
  data={students}              // Array of objects to display
  columns={[
    { key: 'email', label: 'Email' },
    { key: 'name', label: 'Name' },
    { 
      key: 'status', 
      label: 'Status',
      render: (value, row) => <span className="badge">{value}</span>  // Custom rendering
    },
    { key: 'created_at', label: 'Created' }
  ]}
  searchableFields={['email', 'name']}  // Fields to search in
  onRowClick={(row) => console.log('Clicked:', row)}  // Handle row click
/>
```

**Example: Display Students Table**
```jsx
function AdvisorDashboard() {
  const [students, setStudents] = useState([]);

  useEffect(() => {
    // Fetch students from backend
    fetch('/api/v2/students')
      .then(r => r.json())
      .then(data => setStudents(data.students));
  }, []);

  return (
    <DataTable
      data={students}
      columns={[
        { key: 'name', label: '👤 Name' },
        { key: 'email', label: '📧 Email' },
        { 
          key: 'status', 
          label: '📊 Status',
          render: (status) => (
            <span style={{
              background: status === 'Complete' ? '#28a745' : '#ffc107',
              color: 'white',
              padding: '4px 8px',
              borderRadius: '3px'
            }}>
              {status}
            </span>
          )
        },
        { 
          key: 'created_at', 
          label: '📅 Created',
          render: (date) => new Date(date * 1000).toLocaleDateString()
        }
      ]}
      searchableFields={['name', 'email']}
      onRowClick={(student) => viewStudentDetail(student.email)}
    />
  );
}
```

---

### 2. DataValidationBadge Component

**Purpose:** Show data quality score and validation issues.

**Props:**
```jsx
<DataValidationBadge
  data={{
    name: 'John Doe',
    email: 'john@example.com',
    phone: '+1234567890'
  }}
  requiredFields={['name', 'email']}
  validationRules={{
    name: { required: true, minLength: 2, maxLength: 50 },
    email: { required: true, type: 'email' },
    phone: { required: false, type: 'phone' }
  }}
/>
```

**Example: Show Profile Validation**
```jsx
function StudentProfile({ profile }) {
  return (
    <div>
      <h2>My Profile</h2>
      
      <DataValidationBadge
        data={profile}
        requiredFields={['full_name', 'email', 'date_of_birth']}
        validationRules={{
          full_name: { 
            required: true, 
            minLength: 2,
            maxLength: 100 
          },
          email: { 
            required: true, 
            type: 'email' 
          },
          date_of_birth: { 
            required: true,
            type: 'date'
          },
          gpa: { 
            required: false,
            min: 0,
            max: 4
          }
        }}
      />

      <form>
        <input value={profile.full_name} placeholder="Full Name" />
        <input value={profile.email} placeholder="Email" />
        <input value={profile.date_of_birth} placeholder="Date of Birth" type="date" />
        <input value={profile.gpa} placeholder="GPA" type="number" step="0.01" />
      </form>
    </div>
  );
}
```

**Shows:**
- ✅ Valid / ⚠️ Incomplete / ❌ Error status
- Completion percentage
- List of validation issues
- Visual progress bar

---

### 3. AuditLog Component

**Purpose:** Show timeline of all user actions (VIEW, UPDATE, DELETE, etc.).

**Props:**
```jsx
<AuditLog
  logs={[
    {
      id: 'audit_001',
      timestamp: '2026-03-28T14:23:15Z',
      action: 'VIEW',
      actor_email: 'advisor@email.com',
      actor_role: 'advisor',
      target_email: 'student@email.com',
      details: 'Viewed student profile',
      ip_address: '192.168.1.100',
      changes: null
    },
    {
      id: 'audit_002',
      timestamp: '2026-03-28T14:25:30Z',
      action: 'UPDATE',
      actor_email: 'admin@email.com',
      actor_role: 'admin',
      details: 'Updated user eligibility',
      changes: { 
        elig: [{ old_score: 85 }, { new_score: 87 }]
      }
    }
  ]}
/>
```

**Example: Show Recent Activity**
```jsx
function AdvisorDashboard() {
  const [auditLogs, setAuditLogs] = useState([]);

  useEffect(() => {
    // Fetch audit logs
    fetch('/api/v2/audit-logs?limit=20')
      .then(r => r.json())
      .then(data => setAuditLogs(data.logs));
  }, []);

  return (
    <div>
      <h2>Recent Activity</h2>
      <AuditLog logs={auditLogs} />
    </div>
  );
}
```

**Shows:**
- Color-coded action badges (VIEW=blue, UPDATE=orange, DELETE=red)
- Who did what, when, and to whom
- Changed fields (expandable)
- IP address for security audit

---

### 4. SearchFilter Component

**Purpose:** Multi-criteria search and filtering interface.

**Props:**
```jsx
<SearchFilter
  onFilterChange={(filters) => console.log('Filters:', filters)}
  filterOptions={{
    status: ['All', 'Incomplete', 'Complete', 'Approved', 'Rejected'],
    role: ['All', 'student', 'advisor', 'admin'],
    dateRange: true,
    customFilters: [
      {
        name: 'gpa',
        label: 'GPA Range',
        type: 'range',
        min: 0,
        max: 4
      },
      {
        name: 'university',
        label: 'Target University',
        type: 'select',
        options: ['Oxford', 'Cambridge', 'MIT', 'Stanford']
      }
    ]
  }}
/>
```

**Example: Filter Students**
```jsx
function AdvisorDashboard() {
  const [filters, setFilters] = useState({});
  const [filteredStudents, setFilteredStudents] = useState([]);

  const handleFilterChange = (newFilters) => {
    setFilters(newFilters);
    
    // Fetch filtered data from backend
    const params = new URLSearchParams();
    if (newFilters.searchTerm) params.append('q', newFilters.searchTerm);
    if (newFilters.status !== 'All') params.append('status', newFilters.status);
    if (newFilters.dateFrom) params.append('date_from', newFilters.dateFrom);
    if (newFilters.dateTo) params.append('date_to', newFilters.dateTo);

    fetch(`/api/v2/students/search?${params}`)
      .then(r => r.json())
      .then(data => setFilteredStudents(data.students));
  };

  return (
    <div>
      <SearchFilter
        onFilterChange={handleFilterChange}
        filterOptions={{
          status: ['All', 'Incomplete', 'Pending Review', 'Approved'],
          role: ['All', 'student', 'advisor'],
          dateRange: true,
          customFilters: [
            {
              name: 'english_score',
              label: 'IELTS Score Range',
              type: 'range',
              min: 0,
              max: 9
            }
          ]
        }}
      />
      
      <DataTable
        data={filteredStudents}
        columns={[...]}
        searchableFields={['name', 'email']}
      />
    </div>
  );
}
```

**Features:**
- Quick search across multiple fields
- Status/Role filters
- Date range picker
- Custom filters (range, select, text)
- Clear all filters button

---

### 5. VersionComparison Component

**Purpose:** Show data change history with before/after comparison.

**Props:**
```jsx
<VersionComparison
  versions={[
    {
      version: 1,
      timestamp: '2026-03-28T10:00:00Z',
      action: 'CREATED',
      data: { step: 1, profile: {...} },
      changed_by: 'student@email.com'
    },
    {
      version: 2,
      timestamp: '2026-03-28T14:30:00Z',
      action: 'UPDATED',
      data: { step: 2, profile: {...} },
      changed_by: 'student@email.com',
      changes: { step: [1, 2] }
    }
  ]}
/>
```

**Example: Show Student History**
```jsx
function StudentDetail({ studentEmail }) {
  const [versions, setVersions] = useState([]);

  useEffect(() => {
    // Fetch version history
    fetch(`/api/v2/user/${studentEmail}/versions`)
      .then(r => r.json())
      .then(data => setVersions(data.versions));
  }, [studentEmail]);

  return (
    <div>
      <h2>Application History</h2>
      <VersionComparison versions={versions} />
    </div>
  );
}
```

**Features:**
- Timeline of all versions
- Click to view specific version
- Compare with other versions
- See exactly what changed
- Show who made the change and when

---

## 🎨 Complete Example: Advisor Dashboard

```jsx
import React, { useState, useEffect } from 'react';
import DataTable from './DataTable';
import DataValidationBadge from './DataValidationBadge';
import AuditLog from './AuditLog';
import SearchFilter from './SearchFilter';

function AdvisorDashboard() {
  const [students, setStudents] = useState([]);
  const [auditLogs, setAuditLogs] = useState([]);
  const [selectedStudent, setSelectedStudent] = useState(null);
  const [filters, setFilters] = useState({});

  useEffect(() => {
    // Load students and audit logs
    Promise.all([
      fetch('/api/v2/students').then(r => r.json()),
      fetch('/api/v2/audit-logs?limit=10').then(r => r.json())
    ]).then(([studentData, auditData]) => {
      setStudents(studentData.students);
      setAuditLogs(auditData.logs);
    });
  }, []);

  const handleFilterChange = (newFilters) => {
    setFilters(newFilters);
    // Fetch filtered students from backend
  };

  const columns = [
    { key: 'name', label: '👤 Name' },
    { key: 'email', label: '📧 Email' },
    {
      key: 'step',
      label: '📊 Progress',
      render: (step) => `Step ${step} of 5`
    },
    {
      key: 'validation_score',
      label: '✓ Data Quality',
      render: (score) => (
        <div style={{
          background: score >= 90 ? '#28a745' : score >= 70 ? '#ffc107' : '#dc3545',
          color: 'white',
          padding: '4px 8px',
          borderRadius: '3px',
          display: 'inline-block'
        }}>
          {score}%
        </div>
      )
    },
    {
      key: 'created_at',
      label: '📅 Created',
      render: (date) => new Date(date * 1000).toLocaleDateString()
    }
  ];

  return (
    <div style={{ padding: '20px', maxWidth: '1200px', margin: '0 auto' }}>
      <h1>📚 Advisor Dashboard</h1>

      {/* Filter Section */}
      <SearchFilter
        onFilterChange={handleFilterChange}
        filterOptions={{
          status: ['All', 'Step 1', 'Step 2', 'Step 3', 'Step 4', 'Step 5'],
          dateRange: true,
          customFilters: [
            {
              name: 'data_quality',
              label: 'Data Quality',
              type: 'range',
              min: 0,
              max: 100
            }
          ]
        }}
      />

      {/* Students Table */}
      <div style={{ marginBottom: '40px' }}>
        <h2>All Students ({students.length})</h2>
        <DataTable
          data={students}
          columns={columns}
          searchableFields={['name', 'email']}
          onRowClick={(student) => setSelectedStudent(student)}
        />
      </div>

      {/* Selected Student Detail */}
      {selectedStudent && (
        <div style={{
          background: '#f8f9fa',
          border: '1px solid #dee2e6',
          borderRadius: '6px',
          padding: '20px',
          marginBottom: '40px'
        }}>
          <h2>Student Detail: {selectedStudent.name}</h2>
          
          <DataValidationBadge
            data={selectedStudent}
            requiredFields={['name', 'email', 'step']}
          />

          <p>Email: {selectedStudent.email}</p>
          <p>Current Step: {selectedStudent.step}/5</p>
        </div>
      )}

      {/* Audit Log */}
      <div>
        <h2>Recent Activity</h2>
        <AuditLog logs={auditLogs} />
      </div>
    </div>
  );
}

export default AdvisorDashboard;
```

---

## 🔌 Backend API Endpoints Needed

To use these components effectively, implement these backend endpoints:

```
GET /api/v2/students                 - Get all students
GET /api/v2/students/search          - Search/filter students
GET /api/v2/students/{email}         - Get student details
GET /api/v2/students/{email}/profile - Get student profile with validation

GET /api/v2/audit-logs               - Get audit logs
GET /api/v2/audit-logs/user/{email}  - Get logs for specific user

GET /api/v2/user/{email}/versions    - Get version history
GET /api/v2/user/{email}/version/{n} - Get specific version

GET /api/v2/validate-data/{email}    - Get validation status
```

---

## 🚀 Implementation Checklist

- [ ] Copy all 5 components to your project
- [ ] Import components in your pages
- [ ] Connect to backend API endpoints
- [ ] Style and customize as needed
- [ ] Test with real data
- [ ] Add error handling and loading states
- [ ] Make mobile responsive
- [ ] Performance test with large datasets

---

## 📱 Mobile Responsive

All components are built mobile-first and adjust for smaller screens:

- Tables become scrollable on mobile
- Filters stack vertically
- Modal or side panel for details
- Touch-friendly button sizes

---

## ⚡ Performance Tips

1. **Pagination** - Use DataTable's built-in pagination for large datasets
2. **Lazy Loading** - Load audit logs on demand, not all at once
3. **Memoization** - Wrap component in React.memo() if props don't change
4. **Virtual Scrolling** - For 1000+ rows, use virtualization library

---

## 🔒 Security Notes

1. Always validate filters on backend
2. Never send unfiltered raw data to frontend
3. Log all data access (already in AuditLog)
4. Enforce role-based access control
5. Sanitize user inputs before display

---

Ready to integrate? Start with the **AdvisorDashboard** example above! 🎉
