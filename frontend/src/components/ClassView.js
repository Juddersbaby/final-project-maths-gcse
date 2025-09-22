import React from 'react';

export default function ClassView({ classes = [], onSelectClass }) {
    return (
        <div className="space-y-4">
            <h3 className="text-xl font-semibold">Class View</h3>
            <p className="text-sm text-gray-600">Upon logging in, the teacher is presented with a dashboard showing their classes. From here, they can either view the students in a class or generate a quiz for the entire class at once.</p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {(classes.length ? classes : ['Class A', 'Class B']).map((c) => (
                    <div key={c} className="border rounded p-4 bg-gray-50 flex items-center justify-between">
                        <div>
                            <div className="font-medium">{c}</div>
                            <div className="text-xs text-gray-500">23 students</div>
                        </div>
                        <div className="space-x-2">
                            <button className="px-3 py-1 bg-blue-600 text-white rounded" onClick={() => onSelectClass(c)}>View</button>
                            <button className="px-3 py-1 border rounded">Generate Quiz</button>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    )
}
