import React from 'react';

export default function StudentView({ className = 'Class A', students = [], onBack }) {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-xl font-semibold">Student View</h3>
          <div className="text-sm text-gray-600">After selecting a class, the teacher sees a list of all students in that class.</div>
        </div>
        <div>
          <button onClick={onBack} className="px-3 py-1 rounded border">Back</button>
        </div>
      </div>

      <div className="space-y-2">
        {(students.length ? students : ['S001','S002','S003']).map(s => (
          <div key={s} className="flex items-center justify-between border rounded p-3 bg-white">
            <div>
              <div className="font-medium">{s}</div>
              <div className="text-xs text-gray-500">Last activity: 3 days ago</div>
            </div>
            <div className="space-x-2">
              <button className="px-3 py-1 border rounded">Log Result</button>
              <button className="px-3 py-1 bg-emerald-600 text-white rounded">Generate Quiz</button>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
