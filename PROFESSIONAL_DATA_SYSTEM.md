# 🎯 Professional Data Viewing System - Implementation Plan

## Overview
Building a professional data viewing system with:
- ✅ Role-Based Access (Student/Advisor/Admin)
- ✅ Audit Logging (track all access)
- ✅ Search/Filter/Sort tables
- ✅ Data Validation Indicators
- ✅ Historical View (version tracking)

---

## Backend Architecture Requirements

### 1. Role System
```json
Users database should store:
{
  "users": {
    "student@email.com": {
      "name": "John Doe",
      "email": "student@email.com",
      "password_salt": "...",
      "password_hash": "...",
      "role": "student",
      "created_at": 1711612800,
      "updated_at": 1711612800
    },
    "advisor@email.com": {
      "name": "Dr. Smith",
      "email": "advisor@email.com",
      "password_salt": "...",
      "password_hash": "...",
      "role": "advisor",
      "created_at": 1711612800,
      "updated_at": 1711612800
    },
    "admin@email.com": {
      "name": "Admin User",
      "email": "admin@email.com",
      "password_salt": "...",
      "password_hash": "...",
      "role": "admin",
      "created_at": 1711612800,
      "updated_at": 1711612800
    }
  }
}
```

### 2. Audit Logging
```json
{
  "audit_logs": [
    {
      "id": "audit_001",
      "timestamp": "2026-03-28T14:23:15Z",
      "action": "VIEW",
      "actor_email": "advisor@email.com",
      "actor_role": "advisor",
      "target_type": "student_profile",
      "target_email": "student@email.com",
      "details": "Viewed student profile and eligibility",
      "ip_address": "192.168.1.100",
      "changes": null
    },
    {
      "id": "audit_002",
      "timestamp": "2026-03-28T14:25:30Z",
      "action": "UPDATE",
      "actor_email": "admin@email.com",
      "actor_role": "admin",
      "target_type": "user_role",
      "target_email": "student@email.com",
      "details": "Changed user role from student to advisor",
      "changes": {
        "role": ["student", "advisor"]
      }
    }
  ]
}
```

### 3. Version History (Track Changes)
```json
{
  "student@email.com_versions": [
    {
      "version": 1,
      "timestamp": "2026-03-28T10:00:00Z",
      "action": "CREATED",
      "data": {
        "step": 1,
        "profile": {...},
        "docData": {},
        "elig": {}
      },
      "changed_by": "student@email.com"
    },
    {
      "version": 2,
      "timestamp": "2026-03-28T14:30:00Z",
      "action": "UPDATED",
      "data": {
        "step": 2,
        "profile": {...},
        "docData": {...},
        "elig": {}
      },
      "changed_by": "student@email.com",
      "changes": {
        "step": [1, 2],
        "docData": [{}, {...}]
      }
    }
  ]
}
```

### 4. Data Validation Rules
```python
VALIDATION_RULES = {
  "profile": {
    "full_name": {"required": True, "type": "string", "min_length": 2},
    "email": {"required": True, "type": "email"},
    "phone": {"required": False, "type": "phone"},
    "date_of_birth": {"required": False, "type": "date", "format": "YYYY-MM-DD"}
  },
  "documents": {
    "passport": {"required": True, "type": "file"},
    "ielts": {"required": False, "type": "file"}
  },
  "eligibility": {
    "gpa": {"required": False, "type": "float", "min": 0, "max": 4},
    "english_score": {"required": False, "type": "float", "min": 0, "max": 9}
  }
}
```

---

## Backend API Endpoints to Implement

### Audit & Compliance
```
GET  /api/v2/audit-logs             - Get all audit logs (admin only)
GET  /api/v2/audit-logs/user/{email} - Get logs for specific user
POST /api/v2/audit-log              - Log an action (internal)
```

### Role Management
```
GET  /api/v2/users                  - List all users with roles (admin only)
POST /api/v2/users/{email}/role     - Change user role (admin only)
GET  /api/v2/users/{email}/role     - Get user role
```

### Search & Filter
```
GET  /api/v2/students/search?q=...&status=...&step=... - Search students
GET  /api/v2/students/status/summary                 - Count by status
```

### Data Validation
```
GET  /api/v2/validate-data/{email}  - Check data quality
GET  /api/v2/data-quality-report    - Overall system quality (admin)
```

### Version History
```
GET  /api/v2/user/{email}/versions  - Get all versions
GET  /api/v2/user/{email}/version/{n} - Get specific version
```

### Dashboard Data
```
GET  /api/v2/dashboard/student      - Student dashboard (quick stats)
GET  /api/v2/dashboard/advisor      - Advisor dashboard (all students)
GET  /api/v2/dashboard/admin        - Admin dashboard (system metrics)
```

---

## Frontend Components to Build (React)

### 1. Role-Based Dashboards

**StudentDashboard.jsx**
- Personal profile view
- My documents status
- My eligibility
- My chat history (last 5)
- Quick stats (completion %, last updated)

**AdvisorDashboard.jsx**
- Table of all students
- Search/Filter/Sort
- Student detail view (modal)
- Add notes/feedback
- Data quality indicator
- Recent audit log (who accessed what)

**AdminDashboard.jsx**
- System statistics
- User management table
- Role assignment
- Audit log viewer
- Data validation report

### 2. Reusable Components

**DataTable.jsx**
- Headers sortable by click
- Pagination
- Search filter
- Column visibility toggle
- Export to CSV button

**DataValidationBadge.jsx**
- Shows: ✅ Valid / ⚠️ Incomplete / ❌ Error
- Color coded
- Tooltip with details

**AuditLog.jsx**
- Timeline of actions
- Filter by user/action/date
- Color coded actions (VIEW=blue, UPDATE=orange, DELETE=red)

**VersionComparison.jsx**
- Show before/after data
- Highlight what changed
- Show who changed it and when

**SearchFilter.jsx**
- Multi-select status
- Date range picker
- Text search
- Quick filters

---

## Database Schema (If migrating to SQLite later)

```sql
-- Users table
CREATE TABLE users (
  id INTEGER PRIMARY KEY,
  email TEXT UNIQUE NOT NULL,
  name TEXT,
  password_hash TEXT,
  role TEXT DEFAULT 'student',
  created_at TIMESTAMP,
  updated_at TIMESTAMP
);

-- Audit logs table
CREATE TABLE audit_logs (
  id INTEGER PRIMARY KEY,
  timestamp TIMESTAMP,
  action TEXT,
  actor_email TEXT,
  actor_role TEXT,
  target_type TEXT,
  target_email TEXT,
  details TEXT,
  ip_address TEXT,
  FOREIGN KEY (actor_email) REFERENCES users(email)
);

-- Version history table
CREATE TABLE version_history (
  id INTEGER PRIMARY KEY,
  user_email TEXT,
  version INT,
  timestamp TIMESTAMP,
  action TEXT,
  data JSON,
  changed_by TEXT,
  FOREIGN KEY (user_email) REFERENCES users(email),
  FOREIGN KEY (changed_by) REFERENCES users(email)
);

-- Data validation cache
CREATE TABLE validation_results (
  id INTEGER PRIMARY KEY,
  user_email TEXT,
  validation_timestamp TIMESTAMP,
  score FLOAT,
  issues JSON,
  FOREIGN KEY (user_email) REFERENCES users(email)
);
```

---

## File Structure (Current & New)

```
d:\Multiagent\multiagent\data\
├── users\
│   └── users.json (add role field)
├── audit\                          [NEW]
│   └── audit_logs.json
├── versions\                       [NEW]
│   ├── student1@email.com.json
│   └── student2@email.com.json
├── user_state\
│   ├── student@email.com.json
│   └── advisor@email.com.json
├── chat_history\
│   ├── student@email.com.json
│   └── advisor@email.com.json
└── backups\
    ├── users.json.2026-03-28
    └── audit_logs.json.2026-03-28

frontend/
├── pages/
│   ├── StudentDashboard.jsx        [NEW]
│   ├── AdvisorDashboard.jsx        [NEW]
│   ├── AdminDashboard.jsx          [NEW]
│   └── DataHistory.jsx             [NEW]
└── components/
    ├── DataTable.jsx               [NEW]
    ├── DataValidationBadge.jsx     [NEW]
    ├── AuditLog.jsx                [NEW]
    ├── SearchFilter.jsx            [NEW]
    └── VersionComparison.jsx       [NEW]
```

---

## Implementation Steps

### Step 1: Backend Foundation
- [ ] Add role field to users during registration (default: "student")
- [ ] Create audit logging function
- [ ] Create version history tracking
- [ ] Create data validation function
- [ ] Add audit log API endpoints
- [ ] Add version history endpoints
- [ ] Add search/filter endpoints

### Step 2: Frontend Components
- [ ] Create DataTable component with sort/filter
- [ ] Create DataValidationBadge component
- [ ] Create SearchFilter component
- [ ] Create AuditLog component
- [ ] Create StudentDashboard page
- [ ] Create AdvisorDashboard page
- [ ] Create AdminDashboard page

### Step 3: Integration
- [ ] Wire dashboards to backend APIs
- [ ] Add role checks UI (hide admin features from students)
- [ ] Implement auto-logging of all data access
- [ ] Add version history comparison UI

### Step 4: Polish
- [ ] Add loading states
- [ ] Add error handling
- [ ] Add success notifications
- [ ] Mobile responsive design
- [ ] Performance optimization

---

## Security Considerations

1. **Role Validation**
   - Always check role on backend before returning data
   - Students can only see their own data
   - Advisors can see students in their pool
   - Only admins can see system-wide data

2. **Audit Logging**
   - Log ALL data access (reads too, not just writes)
   - Include IP address for compliance
   - Make logs immutable (append-only)

3. **Data Validation**
   - Validate on both frontend and backend
   - Show validation errors to user
   - Log validation failures

4. **Token-Based Auth**
   - Verify token before allowing access
   - Include role in token payload (for frontend decisions)
   - Enforce role on each endpoint

---

## Next Steps

Ready to implement? Start with:

1. **Backend first**: Add role system + audit logging + validation
2. **Frontend second**: Build dashboards + components
3. **Integration**: Wire them together
4. **Testing**: Test as advisor viewing student, admin viewing all, etc.

Let me know which piece you want me to build first! 🚀
