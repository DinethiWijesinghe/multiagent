import React, { useState, useEffect, useCallback } from 'react';
import { apiFetch, apiJson, apiErrorMessage } from './apiClient';

export default function AdminDashboard({ user }) {
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedUser, setSelectedUser] = useState(null);
  const [newRole, setNewRole] = useState('student');

  // Live data state
  const [stats, setStats] = useState(null);
  const [users, setUsers] = useState([]);
  const [students, setStudents] = useState([]);        // for validation tab
  const [auditEvents, setAuditEvents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [roleUpdating, setRoleUpdating] = useState(false);
  const [roleError, setRoleError] = useState(null);

  // Applications tab state
  const [allApps, setAllApps] = useState([]);
  const [appsLoading, setAppsLoading] = useState(false);
  const [appsError, setAppsError] = useState('');
  const [appsFilter, setAppsFilter] = useState('all');
  const [appsBusy, setAppsBusy] = useState('');

  const APP_STATUS_OPTIONS = ['submitted','under_review','accepted','rejected','withdrawn'];
  const APP_STATUS_LABELS = { submitted:'Submitted', under_review:'Under Review', accepted:'Accepted', rejected:'Rejected', withdrawn:'Withdrawn' };
  const APP_STATUS_COLORS = {
    submitted: { bg:'var(--teal-dim)', text:'var(--teal)' },
    under_review: { bg:'var(--amber-dim)', text:'var(--amber)' },
    accepted: { bg:'var(--green-dim)', text:'var(--green)' },
    rejected: { bg:'var(--red-dim)', text:'var(--red)' },
    withdrawn: { bg:'var(--bg3)', text:'var(--text3)' },
  };

  const loadApplications = useCallback(() => {
    setAppsLoading(true);
    setAppsError('');
    apiJson('/applications', user?.token)
      .then(data => setAllApps(Array.isArray(data.applications) ? data.applications : []))
      .catch(e => setAppsError(e.message))
      .finally(() => setAppsLoading(false));
  }, [user?.token]);

  useEffect(() => { if (activeTab === 'applications') loadApplications(); }, [activeTab, loadApplications]);

  const handleAppStatus = async (appId, newStatus) => {
    setAppsBusy(appId);
    try {
      const res = await apiFetch(`/applications/${appId}/status`, user?.token, {
        method: 'PATCH',
        body: JSON.stringify({ status: newStatus }),
      });
      if (!res.ok) throw new Error(await apiErrorMessage(res));
      setAllApps(prev => prev.map(a => a.application_id === appId ? { ...a, status: newStatus } : a));
    } catch(e) {
      setAppsError(e.message);
    } finally {
      setAppsBusy('');
    }
  };

  const loadData = useCallback(() => {
    setLoading(true);
    setError(null);
    Promise.all([
      apiJson('/admin/stats', user?.token).catch((e) => { throw new Error(`Stats error: ${e.message}`); }),
      apiJson('/admin/users', user?.token).catch((e) => { throw new Error(`Users error: ${e.message}`); }),
      apiFetch('/advisor/students', user?.token).then(r => r.ok ? r.json() : Promise.resolve({ students: [] })),
      apiFetch('/admin/audit', user?.token).then(r => r.ok ? r.json() : Promise.resolve({ events: [] })),
    ])
      .then(([statsData, usersData, studentsData, auditData]) => {
        setStats(statsData);
        setUsers(Array.isArray(usersData.users) ? usersData.users : []);
        setStudents(Array.isArray(studentsData.students) ? studentsData.students : []);
        setAuditEvents(Array.isArray(auditData.events) ? auditData.events : []);
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, [user?.token]);

  useEffect(() => { loadData(); }, [loadData]);

  // Derive validation issues from real student data
  const validationIssues = students.flatMap(s => {
    const issues = [];
    if (s.documents === 0) issues.push({ email: s.email, issue: 'No documents uploaded', severity: 'high' });
    else if (s.step < 2) issues.push({ email: s.email, issue: 'Profile incomplete', severity: 'medium' });
    if (s.eligibility === 'ineligible') issues.push({ email: s.email, issue: 'Eligibility check failed', severity: 'high' });
    return issues;
  });
  const getRoleColor = (role) => {
    switch (role) {
      case 'student':
        return { bg: 'var(--accent-dim)', text: 'var(--accent)' };
      case 'advisor':
        return { bg: 'var(--green-dim)', text: 'var(--green)' };
      case 'admin':
        return { bg: 'var(--red-dim)', text: 'var(--red)' };
      default:
        return { bg: 'var(--bg3)', text: 'var(--text3)' };
    }
  };

  const systemStats = stats || { totalUsers: '—', totalStudents: '—', totalAdvisors: '—', totalAdmins: '—', completedApplications: '—', pendingApplications: '—', dataQualityScore: '—' };

  return (
    <div className="fade-up">
      {/* Header */}
      <div className="panel" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <div className="panel-title">⚙️ Admin Dashboard</div>
          <div className="panel-sub">System administration and monitoring</div>
        </div>
        <button className="btn btn-ghost" style={{ fontSize: '0.8rem' }} onClick={loadData} disabled={loading}>
          ↻ Refresh
        </button>
      </div>

      {loading && (
        <div style={{ padding: '1rem', textAlign: 'center', color: 'var(--text3)', fontFamily: 'var(--mono)', fontSize: '0.8rem' }}>
          Loading data…
        </div>
      )}
      {!loading && error && (
        <div style={{ padding: '1rem', background: 'var(--red-dim)', color: 'var(--red)', borderRadius: 'var(--r)', marginBottom: '1rem', fontSize: '0.85rem' }}>
          ⚠ {error}
        </div>
      )}

      {/* System Stats */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', marginBottom: '1.5rem' }}>
        <div className="panel" style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--accent)' }}>{systemStats.totalUsers}</div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: '0.65rem', color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '1px' }}>
            Total Users
          </div>
        </div>
        <div className="panel" style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--green)' }}>{systemStats.completedApplications}</div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: '0.65rem', color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '1px' }}>
            Completed
          </div>
        </div>
        <div className="panel" style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--amber)' }}>{systemStats.pendingApplications}</div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: '0.65rem', color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '1px' }}>
            Pending
          </div>
        </div>
        <div className="panel" style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--accent2)' }}>{systemStats.dataQualityScore}{typeof systemStats.dataQualityScore === 'number' ? '%' : ''}</div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: '0.65rem', color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '1px' }}>
            Data Quality
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1.5rem', borderBottom: '1px solid var(--border)', paddingBottom: '0.5rem' }}>
        {['overview', 'users', 'audit', 'validation', 'applications'].map((tab) => (
          <button
            key={tab}
            className={`tab ${activeTab === tab ? 'active' : ''}`}
            onClick={() => setActiveTab(tab)}
            style={{
              padding: '0.6rem 1.25rem',
              borderBottom: activeTab === tab ? '2px solid var(--accent)' : 'none',
              marginBottom: activeTab === tab ? '-1px' : '0',
              color: activeTab === tab ? 'var(--accent)' : 'var(--text3)',
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              fontFamily: 'var(--sans)',
              fontSize: '0.88rem',
              fontWeight: 600,
              textTransform: 'capitalize',
            }}
          >
            {tab === 'overview' && '📊 Overview'}
            {tab === 'users' && '👥 Users'}
            {tab === 'audit' && '📋 Audit Log'}
            {tab === 'validation' && '✓ Validation'}
            {tab === 'applications' && '📨 Applications'}
          </button>
        ))}
      </div>

      {/* Overview Tab */}
      {activeTab === 'overview' && (
        <div className="fade-up">
          <div className="panel">
            <div className="panel-title">System Metrics</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              <div>
                <div style={{ fontFamily: 'var(--mono)', fontSize: '0.7rem', color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '0.5rem' }}>
                  User Distribution
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                    <span>Students</span>
                    <span style={{ fontWeight: 600, color: 'var(--accent)' }}>{systemStats.totalStudents}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                    <span>Advisors</span>
                    <span style={{ fontWeight: 600, color: 'var(--green)' }}>{systemStats.totalAdvisors}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                    <span>Admins</span>
                    <span style={{ fontWeight: 600, color: 'var(--red)' }}>{systemStats.totalAdmins}</span>
                  </div>
                </div>
              </div>
              <div>
                <div style={{ fontFamily: 'var(--mono)', fontSize: '0.7rem', color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '0.5rem' }}>
                  Application Status
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                    <span>Completed</span>
                    <span style={{ fontWeight: 600, color: 'var(--green)' }}>{systemStats.completedApplications}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                    <span>Pending</span>
                    <span style={{ fontWeight: 600, color: 'var(--amber)' }}>{systemStats.pendingApplications}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                    <span>Validation Issues</span>
                    <span style={{ fontWeight: 600, color: 'var(--red)' }}>{validationIssues.length}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Users Tab */}
      {activeTab === 'users' && (
        <div className="fade-up">
          <div className="panel">
            <div className="panel-title">User Management</div>
            {users.length === 0 && !loading && (
              <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text3)', fontFamily: 'var(--mono)', fontSize: '0.8rem' }}>No users found.</div>
            )}
            {users.length > 0 && (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid var(--border)' }}>
                    {['Name', 'Email', 'Role', 'Registered', 'Action'].map(h => (
                      <th key={h} style={{ textAlign: 'left', padding: '0.75rem', fontFamily: 'var(--mono)', fontSize: '0.65rem', textTransform: 'uppercase', letterSpacing: '1px', color: 'var(--text3)' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {users.map((u) => {
                    const roleColor = getRoleColor(u.role);
                    return (
                      <tr key={u.email} style={{ borderBottom: '1px solid var(--border)' }}>
                        <td style={{ padding: '0.75rem', fontSize: '0.85rem', fontWeight: 600 }}>{u.name}</td>
                        <td style={{ padding: '0.75rem', fontSize: '0.85rem', color: 'var(--text3)' }}>{u.email}</td>
                        <td style={{ padding: '0.75rem' }}>
                          <span style={{ display: 'inline-block', padding: '0.3rem 0.6rem', borderRadius: '4px', background: roleColor.bg, color: roleColor.text, fontSize: '0.75rem', fontWeight: 600, textTransform: 'capitalize' }}>
                            {u.role}
                          </span>
                        </td>
                        <td style={{ padding: '0.75rem', fontSize: '0.85rem', color: 'var(--text3)' }}>{u.created || '—'}</td>
                        <td style={{ padding: '0.75rem' }}>
                          <button
                            className="btn btn-primary"
                            style={{ padding: '0.35rem 0.875rem', fontSize: '0.75rem' }}
                            onClick={() => { setSelectedUser(u); setNewRole(u.role); setRoleError(null); }}
                          >
                            Edit
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            )}
          </div>
        </div>
      )}

      {/* Audit Log Tab */}
      {activeTab === 'audit' && (
        <div className="fade-up">
          <div className="panel">
            <div className="panel-title">Audit Log</div>
            {auditEvents.length === 0 && (
              <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text3)', fontFamily: 'var(--mono)', fontSize: '0.8rem', background: 'var(--bg3)', borderRadius: 'var(--r)' }}>
                No audit events found.
              </div>
            )}
            {auditEvents.length > 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                {auditEvents.map((event, idx) => {
                  const color = event.action === 'LOGIN'
                    ? 'var(--accent)'
                    : event.action === 'DOCUMENT_UPLOAD'
                      ? 'var(--green)'
                      : event.action === 'USER_REGISTER'
                        ? 'var(--amber)'
                        : 'var(--text3)';
                  const icon = event.action === 'LOGIN'
                    ? '🔓'
                    : event.action === 'DOCUMENT_UPLOAD'
                      ? '📄'
                      : event.action === 'USER_REGISTER'
                        ? '🆕'
                        : '🧾';
                  return (
                    <div
                      key={`${event.timestamp || 'ts'}-${event.actor || 'actor'}-${idx}`}
                      style={{
                        padding: '0.875rem',
                        background: 'var(--bg3)',
                        borderRadius: 'var(--r)',
                        border: '1px solid var(--border)',
                        display: 'flex',
                        gap: '1rem',
                      }}
                    >
                      <div style={{ fontSize: '1.25rem', color }}>{icon}</div>
                      <div style={{ flex: 1 }}>
                        <div style={{ fontWeight: 600, color: 'var(--text)', fontSize: '0.88rem' }}>{event.action || 'EVENT'}</div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text3)', marginTop: '0.25rem' }}>
                          {event.details || 'No details'}
                        </div>
                        <div style={{ fontSize: '0.7rem', color: 'var(--text3)', marginTop: '0.25rem' }}>
                          By: {event.actor || 'unknown'}
                        </div>
                      </div>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text3)', whiteSpace: 'nowrap' }}>
                        {event.timestamp ? new Date(event.timestamp).toLocaleString() : '—'}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Applications Tab */}
      {activeTab === 'applications' && (
        <div className="fade-up">
          <div style={{ display: 'flex', gap: '1rem', marginBottom: '1.25rem', flexWrap: 'wrap', alignItems: 'center' }}>
            <select value={appsFilter} onChange={e => setAppsFilter(e.target.value)}
              style={{ padding: '0.6rem 0.875rem', borderRadius: 'var(--r)', border: '1px solid var(--border2)', fontFamily: 'var(--sans)', fontSize: '0.85rem' }}>
              <option value="all">All Statuses</option>
              {APP_STATUS_OPTIONS.map(s => <option key={s} value={s}>{APP_STATUS_LABELS[s]}</option>)}
            </select>
            <button className="btn btn-ghost" onClick={loadApplications} disabled={appsLoading}>↻ Refresh</button>
            <span style={{ fontFamily: 'var(--mono)', fontSize: '0.68rem', color: 'var(--text3)', marginLeft: 'auto' }}>
              {allApps.length} total · {allApps.filter(a=>a.status==='submitted').length} pending
            </span>
          </div>
          {appsError && <div style={{ padding: '0.75rem', background: 'var(--red-dim)', color: 'var(--red)', borderRadius: 'var(--r)', marginBottom: '1rem', fontSize: '0.85rem' }}>⚠ {appsError}</div>}
          {appsLoading && <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text3)', fontFamily: 'var(--mono)', fontSize: '0.8rem' }}>Loading…</div>}
          {!appsLoading && (() => {
            const filtered = appsFilter === 'all' ? allApps : allApps.filter(a => a.status === appsFilter);
            if (!filtered.length) return <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text3)', fontFamily: 'var(--mono)', fontSize: '0.8rem' }}>No applications found.</div>;
            return filtered.map(app => {
              const sc = APP_STATUS_COLORS[app.status] || APP_STATUS_COLORS.submitted;
              const busy = appsBusy === app.application_id;
              return (
                <div key={app.application_id} className="panel" style={{ marginBottom: '0.75rem' }}>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr auto', gap: '1rem', alignItems: 'start' }}>
                    <div>
                      <div style={{ fontWeight: 700, fontSize: '0.92rem' }}>{app.university_name}</div>
                      <div style={{ fontFamily: 'var(--mono)', fontSize: '0.68rem', color: 'var(--text3)', marginTop: '0.2rem' }}>{app.program} · {app.country}</div>
                      <div style={{ fontFamily: 'var(--mono)', fontSize: '0.65rem', color: 'var(--text3)', marginTop: '0.3rem' }}>Student: <strong>{app.user_name || app.user_id}</strong> · {new Date(app.submitted_at).toLocaleDateString()}</div>
                      {app.advisor_notes && <div style={{ marginTop: '0.3rem', fontFamily: 'var(--mono)', fontSize: '0.65rem', color: 'var(--amber)' }}>Note: {app.advisor_notes}</div>}
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', alignItems: 'flex-end' }}>
                      <span style={{ display: 'inline-block', padding: '0.25rem 0.65rem', borderRadius: '4px', background: sc.bg, color: sc.text, fontSize: '0.72rem', fontWeight: 600 }}>
                        {APP_STATUS_LABELS[app.status]}
                      </span>
                      <select disabled={busy} value={app.status} onChange={e => handleAppStatus(app.application_id, e.target.value)}
                        style={{ padding: '0.3rem 0.5rem', borderRadius: 'var(--r)', border: '1px solid var(--border2)', fontFamily: 'var(--mono)', fontSize: '0.7rem', cursor: busy ? 'wait' : 'pointer' }}>
                        {APP_STATUS_OPTIONS.map(s => <option key={s} value={s}>{APP_STATUS_LABELS[s]}</option>)}
                      </select>
                    </div>
                  </div>
                </div>
              );
            });
          })()}
        </div>
      )}

      {/* Validation Tab */}
      {activeTab === 'validation' && (
        <div className="fade-up">
          <div className="panel">
            <div className="panel-title">Data Validation Issues</div>
            {validationIssues.length === 0 && (
              <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--green)', fontFamily: 'var(--mono)', fontSize: '0.8rem' }}>
                ✅ No validation issues found.
              </div>
            )}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              {validationIssues.map((issue, idx) => (
                <div
                  key={idx}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    padding: '0.75rem',
                    background: 'var(--bg3)',
                    borderRadius: 'var(--r)',
                    border: '1px solid var(--border)',
                  }}
                >
                  <div>
                    <div style={{ fontSize: '0.85rem', fontWeight: 600 }}>{issue.email}</div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text3)' }}>{issue.issue}</div>
                  </div>
                  <span
                    style={{
                      display: 'inline-block',
                      padding: '0.3rem 0.6rem',
                      borderRadius: '4px',
                      background: issue.severity === 'high' ? 'var(--red-dim)' : 'var(--amber-dim)',
                      color: issue.severity === 'high' ? 'var(--red)' : 'var(--amber)',
                      fontSize: '0.7rem',
                      fontWeight: 600,
                      textTransform: 'uppercase',
                    }}
                  >
                    {issue.severity}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* User Edit Modal */}
      {selectedUser && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
          }}
          onClick={() => setSelectedUser(null)}
        >
          <div
            className="panel"
            style={{
              maxWidth: '400px',
              width: '90%',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <div className="panel-title">Edit User</div>
              <button
                onClick={() => setSelectedUser(null)}
                style={{
                  background: 'var(--red-dim)',
                  color: 'var(--red)',
                  border: 'none',
                  padding: '0.5rem 0.75rem',
                  borderRadius: 'var(--r)',
                  cursor: 'pointer',
                }}
              >
                ✕
              </button>
            </div>

            <div className="field" style={{ marginBottom: '1rem' }}>
              <label className="flabel">Name</label>
              <input type="text" value={selectedUser.name} disabled style={{ opacity: 0.6 }} />
            </div>

            <div className="field" style={{ marginBottom: '1rem' }}>
              <label className="flabel">Email</label>
              <input type="text" value={selectedUser.email} disabled style={{ opacity: 0.6 }} />
            </div>

            <div className="field" style={{ marginBottom: '1.5rem' }}>
              <label className="flabel">Role</label>
              <select value={newRole} onChange={(e) => setNewRole(e.target.value)}>
                <option value="student">Student</option>
                <option value="advisor">Advisor</option>
                <option value="admin">Admin</option>
              </select>
            </div>

            {roleError && (
              <div style={{ padding: '0.75rem', background: 'var(--red-dim)', color: 'var(--red)', borderRadius: 'var(--r)', marginBottom: '1rem', fontSize: '0.82rem' }}>
                ⚠ {roleError}
              </div>
            )}

            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <button
                className="btn btn-primary"
                style={{ flex: 1 }}
                disabled={roleUpdating}
                onClick={() => {
                  setRoleUpdating(true);
                  setRoleError(null);
                  apiFetch(`/admin/users/${encodeURIComponent(selectedUser.email)}/role`, user?.token, {
                    method: 'PATCH',
                    body: JSON.stringify({ role: newRole }),
                  })
                    .then(async (r) => {
                      if (!r.ok) throw new Error(await apiErrorMessage(r));
                      return r.json();
                    })
                    .then(() => {
                      setUsers(prev => prev.map(u => u.email === selectedUser.email ? { ...u, role: newRole } : u));
                      setSelectedUser(null);
                      loadData();
                    })
                    .catch(e => setRoleError(e.message))
                    .finally(() => setRoleUpdating(false));
                }}
              >
                {roleUpdating ? 'Saving…' : 'Save Changes'}
              </button>
              <button
                className="btn btn-ghost"
                style={{ flex: 1 }}
                onClick={() => { setSelectedUser(null); setRoleError(null); }}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
