import sqlite3, pandas as pd

def load_logs(db='app.db'):
    conn = sqlite3.connect(db)
    logs = pd.read_sql_query('SELECT * FROM ab_log ORDER BY ts', conn)
    attempts = pd.read_sql_query('SELECT * FROM attempts ORDER BY ts', conn)
    conn.close()
    return logs, attempts

def policy_summary(db='app.db'):
    logs, attempts = load_logs(db)
    if logs.empty:
        print('No ab_log entries')
        return None
    # naive heuristic: for each log entry, compute next 3 attempts correctness for that student
    rows = []
    for _, r in logs.iterrows():
        sid = r['student_id']
        ts = pd.to_datetime(r['ts'])
        future = attempts[(attempts['student_id']==sid) & (pd.to_datetime(attempts['ts'])>ts)].head(3)
        if future.empty:
            mean_correct = None
        else:
            mean_correct = future['correct'].astype(float).mean()
        rows.append({'policy': r['policy'], 'student_id': sid, 'mean_next3_correct': mean_correct})
    df = pd.DataFrame(rows)
    return df.groupby('policy')['mean_next3_correct'].agg(['mean','count'])

if __name__ == '__main__':
    print(policy_summary())
