import React, { useState } from 'react';
import { ArrowLeft, BookOpen, Code, Lightbulb, Settings, Brain } from 'lucide-react';
import { OverviewTab, MathTab, CodeTab, PreprocessingTab, TipsTab } from './AlgorithmDetailTabs';

const AlgorithmDetailPage = ({ algorithmData, onBack }) => {
  const [activeTab, setActiveTab] = useState('overview');

  const tabs = [
    { id: 'overview', label: 'Overview', icon: BookOpen, component: OverviewTab, data: algorithmData.overview },
    { id: 'math', label: 'Mathematics', icon: Brain, component: MathTab, data: algorithmData.math },
    { id: 'code', label: 'Code Examples', icon: Code, component: CodeTab, data: algorithmData.code },
    { id: 'preprocessing', label: 'Preprocessing', icon: Settings, component: PreprocessingTab, data: algorithmData.preprocessing },
    { id: 'tips', label: 'Tips & Tricks', icon: Lightbulb, component: TipsTab, data: algorithmData.tips },
  ];

  const currentTab = tabs.find(t => t.id === activeTab);
  const TabComponent = currentTab.component;

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-2xl shadow-2xl p-8 mb-6">
          <button
            onClick={onBack}
            className="flex items-center gap-2 text-indigo-600 hover:text-indigo-700 mb-4 font-medium transition"
          >
            <ArrowLeft className="w-5 h-5" />
            Back to Selector
          </button>
          
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-3 mb-2">
                <h1 className="text-4xl font-bold text-gray-800">{algorithmData.name}</h1>
                <span className="px-3 py-1 bg-indigo-100 text-indigo-700 rounded-full text-sm font-semibold">
                  {algorithmData.category}
                </span>
              </div>
              <p className="text-gray-600 text-lg">{algorithmData.description}</p>
            </div>
          </div>

          {/* Algorithm Type Badges */}
          <div className="mt-4 flex flex-wrap gap-2">
            {algorithmData.badges.map((badge, idx) => (
              <span
                key={idx}
                className={`px-3 py-1 bg-${badge.color}-100 text-${badge.color}-700 rounded-full text-sm`}
              >
                {badge.label}
              </span>
            ))}
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="bg-white rounded-xl shadow-lg mb-6 p-2 flex gap-2 overflow-x-auto">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-3 rounded-lg font-medium transition whitespace-nowrap ${
                  activeTab === tab.id
                    ? 'bg-indigo-600 text-white'
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                <Icon className="w-4 h-4" />
                {tab.label}
              </button>
            );
          })}
        </div>

        {/* Content */}
        <div className="bg-white rounded-2xl shadow-2xl p-8">
          <TabComponent data={currentTab.data} />
        </div>
      </div>
    </div>
  );
};

export default AlgorithmDetailPage;