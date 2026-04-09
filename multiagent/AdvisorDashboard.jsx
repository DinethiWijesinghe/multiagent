import React, { useState, useEffect, useCallback } from 'react';
import { apiJson, apiFetch } from './apiClient';

const APP_STATUS_OPTIONS = ['submitted','under_review','accepted','rejected','withdrawn'];
const APP_STATUS_LABELS = { submitted:'Submitted', under_review:'Under Review', accepted:'Accepted', rejected:'Rejected', withdrawn:'Withdrawn' };
const APP_STATUS_COLORS = {
  submitted: { bg:'var(--teal-dim)', text:'var(--teal)' },
  under_review: { bg:'var(--amber-dim)', text:'var(--amber)' },
  accepted: { bg:'var(--green-dim)', text:'var(--green)' },
  rejected: { bg:'var(--red-dim)', text:'var(--red)' },
  withdrawn: { bg:'var(--bg3)', text:'var(--text3)' },
};

export default function AdvisorDashboard({ user }) {
  const [activeTab, setActiveTab] = useState('students');
  const [students, setStudents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState('all');
  const [selectedStudent, setSelectedStudent] = useState(null);
  const [feedback, setFeedback] = useState('');
  const [saving, setSaving] = useState(false);

  // Applications tab state
  const [apps, setApps] = useState([]);
  const [appsLoading, setAppsLoading] = useState(false);
  const [appsError, setAppsError] = useState('');
  const [appsBusy, setAppsBusy] = useState('');
  const [appsSearch, setAppsSearch] = useState('');
  const [appsFilter, setAppsFilter] = useState('all');

  const loadApplications = useCallback(() => {
    setAppsLoading(true);
    setAppsError('');
    apiJson('/applications', user?.token)
      .then(data => setApps(Array.isArray(data.applications) ? data.applications : []))
      .catch(e => setAppsError(e.message))
      .finally(() => setAppsLoading(false));
  }, [user?.token]);

  useEffect(() => { if (activeTab === 'applications') loadApplications(); }, [activeTab, loadApplications]);

  const handleStatusChange = async (appId, newStatus, notes) => {
    setAppsBusy(appId);
    try {
      const res = await apiFetch(`/applications/${appId}/status`, user?.token, {
        method: 'PATCH',
        body: JSON.stringify({ status: newStatus, advisor_notes: notes || null }),
      });
      if (!res.ok) { const t = await res.text().catch(() => ''); throw new Error(t || 'Update failed'); }
      setApps(prev => prev.map(a => a.application_id === appId ? { ...a, status: newStatus, advisor_notes: notes || a.advisor_notes } : a));
    } catch(e) {
      setAppsError(e.message);
    } finally {
      setAppsBusy('');
    }
  };

  const loadStudents = useCallback(() => {
    setLoading(true);
    setError(null);
    apiJson('/advisor/students', user?.token)
      .then(data => setStudents(Array.isArray(data.students) ? data.students : []))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, [user?.token]);

  useEffect(() => { loadStudents(); }, [loadStudents]);

  const filteredStudents = students.filter((s) => {
    const matchesSearch = s.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      s.email.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = filterStatus === 'all' || s.status === filterStatus;
    return matchesSearch && matchesStatus;
  });

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed':
        return { bg: 'var(--green-dim)', text: 'var(--green)', label: '✅ Completed' };
      case 'in-progress':
        return { bg: 'var(--accent-dim)', text: 'var(--accent)', label: '⚙️ In Progress' };
      case 'started':
        return { bg: 'var(--amber-dim)', text: 'var(--amber)', label: '📝 Started' };
      default:
        return { bg: 'var(--bg3)', text: 'var(--text3)', label: 'Unknown' };
    }
  };

  const getQualityIcon = (quality) => {
    switch (quality) {
      case 'excellent':
        return '⭐⭐⭐';
      case 'good':
        return '⭐⭐';
      case 'warning':
        return '⭐';
      default:
        return '—';
    }
  };

  return (
    <div className="fade-up">
      {/* Header */}
      <div className="panel">
        <div className="panel-title">🎯 Advisor Dashboard</div>
        <div className="panel-sub">Manage and monitor all students</div>
      </div>

      {/* Tabs */}
      <div className="tabs" style={{ marginBottom: '1.5rem' }}>
        <button className={`tab${activeTab === 'students' ? ' active' : ''}`} onClick={() => setActiveTab('students')}>Students</button>
        <button className={`tab${activeTab === 'applications' ? ' active' : ''}`} onClick={() => setActiveTab('applications')}>
          Applications {apps.filter(a => a.status === 'submitted').length > 0 && <span style={{ marginLeft: '.4rem', background: 'var(--accent)', color: '#fff', borderRadius: '10px', padding: '0 .45rem', fontSize: '.65rem', fontWeight: 700 }}>{apps.filter(a => a.status === 'submitted').length}</span>}
        </button>
      </div>

      {/* ── Applications Tab ── */}
      {activeTab === 'applications' && (
        <div>
          <div style={{ display: 'flex', gap: '1rem', marginBottom: '1.25rem', flexWrap: 'wrap' }}>
            <input type="text" placeholder="Search student or university…" value={appsSearch} onChange={e => setAppsSearch(e.target.value)}
              style={{ flex: 1, minWidth: 0, padding: '0.6rem 0.875rem', borderRadius: 'var(--r)', border: '1px solid var(--border2)', fontFamily: 'var(--sans)', fontSize: '0.88rem' }} />
            <select value={appsFilter} onChange={e => setAppsFilter(e.target.value)}
              style={{ padding: '0.6rem 0.875rem', borderRadius: 'var(--r)', border: '1px solid var(--border2)', fontFamily: 'var(--sans)', fontSize: '0.88rem' }}>
              <option value="all">All Statuses</option>
              {APP_STATUS_OPTIONS.map(s => <option key={s} value={s}>{APP_STATUS_LABELS[s]}</option>)}
            </select>
            <button className="btn btn-ghost" onClick={loadApplications} disabled={appsLoading}>↻ Refresh</button>
          </div>
          {appsError && <div style={{ padding: '0.75rem', background: 'var(--red-dim)', color: 'var(--red)', borderRadius: 'var(--r)', marginBottom: '1rem', fontSize: '0.85rem' }}>⚠ {appsError}</div>}
          {appsLoading && <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text3)', fontFamily: 'var(--mono)', fontSize: '0.8rem' }}>Loading applications…</div>}
          {!appsLoading && (() => {
            const filtered = apps.filter(a => {
              const matchSearch = a.university_name?.toLowerCase().includes(appsSearch.toLowerCase()) || a.user_name?.toLowerCase().includes(appsSearch.toLowerCase()) || a.user_id?.toLowerCase().includes(appsSearch.toLowerCase());
              const matchFilter = appsFilter === 'all' || a.status === appsFilter;
              return matchSearch && matchFilter;
            });
            if (!filtered.length) return <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text3)', fontFamily: 'var(--mono)', fontSize: '0.8rem' }}>No applications found.</div>;
            return filtered.map(app => {
              const sc = APP_STATUS_COLORS[app.status] || APP_STATUS_COLORS.submitted;
              const busy = appsBusy === app.application_id;
              return (
                <div key={app.application_id} className="panel" style={{ marginBottom: '0.75rem' }}>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr auto', gap: '1rem', alignItems: 'start' }}>
                    <div>
                      <div style={{ fontWeight: 700, fontSize: '0.95rem' }}>{app.university_name}</div>
                      <div style={{ fontFamily: 'var(--mono)', fontSize: '0.7rem', color: 'var(--text3)', marginTop: '0.2rem' }}>
                        {app.program} · {app.country}
                      </div>
                      <div style={{ fontFamily: 'var(--mono)', fontSize: '0.65rem', color: 'var(--text3)', marginTop: '0.35rem' }}>
                        Student: <strong>{app.user_name || app.user_id}</strong> · Submitted {new Date(app.submitted_at).toLocaleDateString()}
                      </div>
                      {app.advisor_notes && <div style={{ marginTop: '0.35rem', fontFamily: 'var(--mono)', fontSize: '0.67rem', color: 'var(--amber)' }}>Note: {app.advisor_notes}</div>}
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', alignItems: 'flex-end' }}>
                      <span style={{ display: 'inline-block', padding: '0.25rem 0.65rem', borderRadius: '4px', background: sc.bg, color: sc.text, fontSize: '0.75rem', fontWeight: 600 }}>
                        {APP_STATUS_LABELS[app.status]}
                      </span>
                      <select
                        disabled={busy}
                        value={app.status}
                        onChange={e => handleStatusChange(app.application_id, e.target.value, app.advisor_notes)}
                        style={{ padding: '0.3rem 0.5rem', borderRadius: 'var(--r)', border: '1px solid var(--border2)', fontFamily: 'var(--mono)', fontSize: '0.7rem', cursor: busy ? 'wait' : 'pointer' }}
                      >
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

      {/* ── Students Tab ── */}
      {activeTab === 'students' && <>

      {/* Empty-state banner when no students exist */}
      {!loading && !error && students.length === 0 && (
        <div style={{ padding: '1.25rem', background: 'var(--bg3)', border: '1px solid var(--border2)', borderRadius: 'var(--r)', marginBottom: '1.25rem', fontSize: '0.85rem', color: 'var(--text3)', fontFamily: 'var(--mono)', lineHeight: 1.6 }}>
          ℹ No student accounts yet. Student data appears here once students register and begin their applications.
          <div style={{ marginTop: '0.5rem', fontSize: '0.75rem' }}>
            Demo: log in as <strong>student@example.com</strong> / <strong>Student@123</strong> and complete the profile wizard.
          </div>
        </div>
      )}

      {/* Stats */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', marginBottom: '1.5rem' }}>
        <div className="panel" style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--accent)' }}>{students.length}</div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: '0.65rem', color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '1px' }}>
            Total Students
          </div>
        </div>
        <div className="panel" style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--green)' }}>
            {students.filter((s) => s.status === 'completed').length}
          </div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: '0.65rem', color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '1px' }}>
            Completed
          </div>
        </div>
        <div className="panel" style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--accent2)' }}>
            {students.filter((s) => s.status === 'in-progress').length}
          </div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: '0.65rem', color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '1px' }}>
            In Progress
          </div>
        </div>
        <div className="panel" style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--amber)' }}>
            {students.filter((s) => s.eligibility === 'ineligible').length}
          </div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: '0.65rem', color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '1px' }}>
            Need Support
          </div>
        </div>
      </div>

      {/* Search & Filter */}
      <div className="panel" style={{ display: 'flex', gap: '1rem', marginBottom: '1.5rem' }}>
        <input
          type="text"
          placeholder="Search by name or email..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          style={{ flex: 1, padding: '0.6rem 0.875rem', borderRadius: 'var(--r)', border: '1px solid var(--border2)', fontFamily: 'var(--sans)', fontSize: '0.88rem' }}
        />
        <select
          value={filterStatus}
          onChange={(e) => setFilterStatus(e.target.value)}
          style={{ padding: '0.6rem 0.875rem', borderRadius: 'var(--r)', border: '1px solid var(--border2)', fontFamily: 'var(--sans)', fontSize: '0.88rem', minWidth: '150px' }}
        >
          <option value="all">All Status</option>
          <option value="completed">Completed</option>
          <option value="in-progress">In Progress</option>
          <option value="started">Started</option>
        </select>
        <button
          className="btn btn-ghost"
          style={{ fontSize: '0.8rem', whiteSpace: 'nowrap' }}
          onClick={loadStudents}
          disabled={loading}
        >
          ↻ Refresh
        </button>
      </div>

      {/* Students Table */}
      <div className="panel">
        <div className="panel-title">📋 Students</div>

        {loading && (
          <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text3)', fontFamily: 'var(--mono)', fontSize: '0.8rem' }}>
            Loading students…
          </div>
        )}

        {!loading && error && (
          <div style={{ padding: '1rem', background: 'var(--red-dim)', color: 'var(--red)', borderRadius: 'var(--r)', marginBottom: '1rem', fontSize: '0.85rem' }}>
            ⚠ {error}
          </div>
        )}

        {!loading && !error && filteredStudents.length === 0 && (
          <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text3)', fontFamily: 'var(--mono)', fontSize: '0.8rem', lineHeight: 1.7 }}>
            {students.length === 0
              ? <>No student accounts registered yet.<br/><span style={{fontSize:'.72rem'}}>Students will appear here after they sign up and start the application wizard.</span></>
              : 'No students match your filter.'}
          </div>
        )}

        {!loading && filteredStudents.length > 0 && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid var(--border)' }}>
                {['Name', 'Progress', 'Status', 'Data Quality', 'Action'].map(h => (
                  <th key={h} style={{ textAlign: 'left', padding: '0.75rem', fontFamily: 'var(--mono)', fontSize: '0.65rem', textTransform: 'uppercase', letterSpacing: '1px', color: 'var(--text3)' }}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filteredStudents.map((student) => {
                const statusColor = getStatusColor(student.status);
                return (
                  <tr key={student.email} style={{ borderBottom: '1px solid var(--border)' }}>
                    <td style={{ padding: '0.75rem', fontSize: '0.85rem' }}>
                      <div style={{ fontWeight: 600 }}>{student.name}</div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text3)' }}>{student.email}</div>
                    </td>
                    <td style={{ padding: '0.75rem' }}>
                      <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--accent)' }}>{student.completion}%</div>
                      <div style={{ height: '3px', background: 'var(--border)', borderRadius: '2px', marginTop: '0.3rem', overflow: 'hidden' }}>
                        <div
                          style={{
                            height: '100%',
                            background: 'linear-gradient(90deg, var(--accent), var(--green))',
                            width: `${student.completion}%`,
                          }}
                        />
                      </div>
                    </td>
                    <td style={{ padding: '0.75rem' }}>
                      <span
                        style={{
                          display: 'inline-block',
                          padding: '0.3rem 0.6rem',
                          borderRadius: '4px',
                          background: statusColor.bg,
                          color: statusColor.text,
                          fontSize: '0.75rem',
                          fontWeight: 600,
                        }}
                      >
                        {statusColor.label}
                      </span>
                    </td>
                    <td style={{ padding: '0.75rem', fontSize: '0.85rem' }}>
                      <div>{getQualityIcon(student.dataQuality)}</div>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text3)', textTransform: 'capitalize' }}>{student.dataQuality}</div>
                    </td>
                    <td style={{ padding: '0.75rem' }}>
                      <button
                        className="btn btn-primary"
                        style={{ padding: '0.35rem 0.875rem', fontSize: '0.75rem' }}
                        onClick={() => { setSelectedStudent(student); setFeedback(''); }}
                      >
                        View
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

      {/* Student Detail Modal */}
      {selectedStudent && (
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
          onClick={() => setSelectedStudent(null)}
        >
          <div
            className="panel"
            style={{
              maxWidth: '500px',
              width: '90%',
              maxHeight: '80vh',
              overflowY: 'auto',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <div className="panel-title">👤 {selectedStudent.name}</div>
              <button
                onClick={() => setSelectedStudent(null)}
                style={{
                  background: 'var(--red-dim)',
                  color: 'var(--red)',
                  border: 'none',
                  padding: '0.5rem 0.75rem',
                  borderRadius: 'var(--r)',
                  cursor: 'pointer',
                  fontSize: '1rem',
                }}
              >
                ✕
              </button>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', marginBottom: '1rem' }}>
              {[
                ['Email', selectedStudent.email],
                ['Last Updated', selectedStudent.lastUpdated || '—'],
                ['Documents', `${selectedStudent.documents} uploaded`],
                ['Eligibility',
                  selectedStudent.eligibility === 'eligible' ? '✅ Eligible'
                  : selectedStudent.eligibility === 'ineligible' ? '🔴 Ineligible'
                  : '⚠️ Pending'],
                ['Step', `${selectedStudent.step} / 4`],
                ['Completion', `${selectedStudent.completion}%`],
              ].map(([label, val]) => (
                <div key={label}>
                  <div style={{ fontFamily: 'var(--mono)', fontSize: '0.6rem', color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '1px' }}>{label}</div>
                  <div style={{ fontSize: '0.9rem', color: 'var(--text)' }}>{val}</div>
                </div>
              ))}
            </div>

            <div style={{ fontFamily: 'var(--mono)', fontSize: '0.6rem', color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '0.5rem' }}>
              Add Feedback
            </div>
            <textarea
              value={feedback}
              onChange={e => setFeedback(e.target.value)}
              placeholder="Add notes or feedback for this student..."
              style={{
                width: '100%',
                padding: '0.75rem',
                borderRadius: 'var(--r)',
                border: '1px solid var(--border)',
                fontFamily: 'var(--sans)',
                fontSize: '0.85rem',
                minHeight: '80px',
                marginBottom: '1rem',
              }}
            />

            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <button
                className="btn btn-primary"
                style={{ flex: 1 }}
                disabled={saving}
                onClick={() => {
                  setSaving(true);
                  // Feedback persistence endpoint not yet implemented — local close only.
                  setTimeout(() => { setSaving(false); setSelectedStudent(null); }, 400);
                }}
              >
                {saving ? 'Saving…' : 'Save Feedback'}
              </button>
              <button
                className="btn btn-ghost"
                style={{ flex: 1 }}
                onClick={() => setSelectedStudent(null)}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
      </>}
    </div>
  );
}
