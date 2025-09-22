import React from 'react'

export default function Modal({ open, onClose, title, children }) {
    if (!open) return null
    return (
        <div className="fixed inset-0 bg-black bg-opacity-30 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg p-6 w-full max-w-lg">
                <div className="flex items-center justify-between mb-4">
                    <h4 className="font-semibold">{title}</h4>
                    <button onClick={onClose} className="text-gray-500">Close</button>
                </div>
                <div>{children}</div>
            </div>
        </div>
    )
}
