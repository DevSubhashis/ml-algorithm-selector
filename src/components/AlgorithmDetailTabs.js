import React from 'react';

// Tab components for different sections
export const OverviewTab = ({ data }) => {
  return (
    <div className="space-y-8">
      {/* What is section */}
      {data.whatIs && (
        <section>
          <h2 className="text-2xl font-bold text-gray-800 mb-4">{data.whatIs.title}</h2>
          <p className="text-gray-700 leading-relaxed mb-4">{data.whatIs.description}</p>
          {data.whatIs.highlight && (
            <div className="bg-indigo-50 border-l-4 border-indigo-600 p-4 rounded">
              <p className="text-indigo-900 font-medium">{data.whatIs.highlight}</p>
            </div>
          )}
        </section>
      )}

      {/* When to use */}
      {data.whenToUse && (
        <section>
          <h2 className="text-2xl font-bold text-gray-800 mb-4">{data.whenToUse.title}</h2>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <h3 className="font-semibold text-green-900 mb-2">✓ Perfect For:</h3>
              <ul className="space-y-2 text-sm text-gray-700">
                {data.whenToUse.perfectFor.map((item, idx) => (
                  <li key={idx}>• {item}</li>
                ))}
              </ul>
            </div>
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <h3 className="font-semibold text-red-900 mb-2">✗ Avoid When:</h3>
              <ul className="space-y-2 text-sm text-gray-700">
                {data.whenToUse.avoidWhen.map((item, idx) => (
                  <li key={idx}>• {item}</li>
                ))}
              </ul>
            </div>
          </div>
        </section>
      )}

      {/* Use Cases */}
      {data.useCases && (
        <section>
          <h2 className="text-2xl font-bold text-gray-800 mb-4">{data.useCases.title}</h2>
          <div className="grid md:grid-cols-2 gap-4">
            {data.useCases.cases.map((useCase, idx) => (
              <div key={idx} className="bg-blue-50 rounded-lg p-5 border border-blue-200">
                <h3 className="font-semibold text-blue-900 mb-2">{useCase.icon} {useCase.title}</h3>
                <p className="text-sm text-gray-700">{useCase.description}</p>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Pros and Cons */}
      {data.prosAndCons && (
        <section>
          <h2 className="text-2xl font-bold text-gray-800 mb-4">{data.prosAndCons.title}</h2>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-semibold text-green-700 mb-3 text-lg">✓ Advantages</h3>
              <ul className="space-y-3">
                {data.prosAndCons.pros.map((pro, idx) => (
                  <li key={idx} className="flex items-start gap-2">
                    <span className="text-green-600 text-lg">✓</span>
                    <span className="text-gray-700 text-sm">{pro}</span>
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <h3 className="font-semibold text-red-700 mb-3 text-lg">✗ Limitations</h3>
              <ul className="space-y-3">
                {data.prosAndCons.cons.map((con, idx) => (
                  <li key={idx} className="flex items-start gap-2">
                    <span className="text-red-600 text-lg">✗</span>
                    <span className="text-gray-700 text-sm">{con}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </section>
      )}

      {/* Step by Step */}
      {data.stepByStep && (
        <section>
          <h2 className="text-2xl font-bold text-gray-800 mb-4">{data.stepByStep.title}</h2>
          <div className="space-y-4">
            {data.stepByStep.steps.map((item, idx) => (
              <div key={idx} className="flex gap-4 items-start bg-purple-50 p-4 rounded-lg border border-purple-200">
                <div className="flex-shrink-0 w-8 h-8 bg-purple-600 text-white rounded-full flex items-center justify-center font-bold">
                  {idx + 1}
                </div>
                <div>
                  <h3 className="font-semibold text-purple-900 mb-1">{item.title}</h3>
                  <p className="text-sm text-gray-700">{item.description}</p>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
};

export const MathTab = ({ data }) => {
  return (
    <div className="space-y-8">
      {/* Objective Function */}
      {data.objectiveFunction && (
        <div className="bg-gray-50 border-l-4 border-indigo-600 p-6 rounded-lg">
          <h3 className="font-semibold text-lg text-gray-800 mb-3">{data.objectiveFunction.title}</h3>
          <div className="bg-white p-4 rounded border border-gray-200 font-mono text-sm overflow-x-auto">
            <p>{data.objectiveFunction.formula}</p>
          </div>
          {data.objectiveFunction.parameters && (
            <div className="mt-4 space-y-2 text-sm text-gray-700">
              <p><strong>Where:</strong></p>
              <ul className="ml-4 space-y-1">
                {data.objectiveFunction.parameters.map((param, idx) => (
                  <li key={idx}>• {param}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Additional Math Content */}
      {data.additionalContent && data.additionalContent.map((section, idx) => (
        <div key={idx} className="bg-gray-50 border-l-4 border-green-600 p-6 rounded-lg">
          <h3 className="font-semibold text-lg text-gray-800 mb-3">{section.title}</h3>
          {section.formula && (
            <div className="bg-white p-4 rounded border border-gray-200 font-mono text-sm overflow-x-auto">
              <p>{section.formula}</p>
            </div>
          )}
          {section.description && (
            <p className="mt-3 text-sm text-gray-700">{section.description}</p>
          )}
        </div>
      ))}

      {/* Visualization */}
      {data.visualization && (
        <div className="bg-gradient-to-br from-indigo-50 to-purple-50 p-6 rounded-lg border border-indigo-200">
          <h3 className="font-semibold text-indigo-900 mb-4">{data.visualization.title}</h3>
          <div className="grid md:grid-cols-3 gap-4 mb-4">
            {data.visualization.items.map((item, idx) => (
              <div key={idx} className="bg-white p-4 rounded border-2" style={{borderColor: item.color}}>
                <h4 className="font-semibold mb-2" style={{color: item.color}}>{item.title}</h4>
                <p className="text-xs text-gray-700">{item.description}</p>
              </div>
            ))}
          </div>
          {data.visualization.insight && (
            <p className="text-sm text-gray-700">
              <strong>Key Insight:</strong> {data.visualization.insight}
            </p>
          )}
        </div>
      )}
    </div>
  );
};

export const CodeTab = ({ data }) => {
  return (
    <div className="space-y-6">
      {data.examples && data.examples.map((example, idx) => (
        <div key={idx} className="mb-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-3">{example.title}</h3>
          <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
            <pre className="text-sm"><code>{example.code}</code></pre>
          </div>
        </div>
      ))}
    </div>
  );
};

export const PreprocessingTab = ({ data }) => {
  return (
    <div className="space-y-6">
      {/* Critical Warning */}
      {data.critical && (
        <div className="bg-red-50 border-l-4 border-red-600 p-6 rounded-lg">
          <h3 className="text-lg font-semibold text-red-900 mb-3">🚨 {data.critical.title}</h3>
          <p className="text-gray-700 mb-4">{data.critical.description}</p>
          {data.critical.code && (
            <div className="bg-white p-4 rounded border border-red-200">
              <pre className="text-sm"><code>{data.critical.code}</code></pre>
            </div>
          )}
          {data.critical.why && (
            <p className="text-sm text-gray-700 mt-3">
              <strong>Why?</strong> {data.critical.why}
            </p>
          )}
        </div>
      )}

      {/* Steps */}
      {data.steps && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-800">{data.stepsTitle}</h3>
          {data.steps.map((step, idx) => (
            <div key={idx} className="bg-blue-50 p-5 rounded-lg border border-blue-200">
              <h4 className="font-semibold text-blue-900 mb-2">{idx + 1}. {step.title}</h4>
              <p className="text-sm text-gray-700 mb-3">{step.description}</p>
              {step.code && (
                <div className="bg-white p-3 rounded text-xs">
                  <code>{step.code}</code>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Complete Pipeline */}
      {data.completePipeline && (
        <div className="bg-gradient-to-br from-purple-50 to-indigo-50 p-6 rounded-lg border border-purple-200">
          <h3 className="text-lg font-semibold text-purple-900 mb-3">{data.completePipeline.title}</h3>
          <div className="bg-white p-4 rounded">
            <pre className="text-sm overflow-x-auto"><code>{data.completePipeline.code}</code></pre>
          </div>
        </div>
      )}

      {/* Common Mistakes */}
      {data.mistakes && (
        <div className="bg-yellow-50 border-l-4 border-yellow-600 p-6 rounded-lg">
          <h3 className="text-lg font-semibold text-yellow-900 mb-3">⚠️ {data.mistakes.title}</h3>
          <ul className="space-y-2 text-sm text-gray-700">
            {data.mistakes.items.map((item, idx) => (
              <li key={idx} className="flex items-start gap-2">
                <span className="text-red-600 font-bold">✗</span>
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export const TipsTab = ({ data }) => {
  return (
    <div className="space-y-6">
      {/* Hyperparameter Tuning */}
      {data.hyperparameterTuning && (
        <div className="bg-gradient-to-br from-green-50 to-emerald-50 p-6 rounded-lg border border-green-200">
          <h3 className="text-lg font-semibold text-green-900 mb-3">{data.hyperparameterTuning.title}</h3>
          <div className="space-y-4">
            {data.hyperparameterTuning.sections.map((section, idx) => (
              <div key={idx}>
                <h4 className="font-semibold text-green-800 mb-2">{section.title}</h4>
                <ul className="ml-4 space-y-1 text-sm text-gray-700">
                  {section.points.map((point, pidx) => (
                    <li key={pidx}>• {point}</li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Best Practices */}
      {data.bestPractices && (
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-blue-50 p-5 rounded-lg border border-blue-200">
            <h4 className="font-semibold text-blue-900 mb-3">✓ Best Practices</h4>
            <ul className="space-y-2 text-sm text-gray-700">
              {data.bestPractices.dos.map((item, idx) => (
                <li key={idx}>• {item}</li>
              ))}
            </ul>
          </div>
          <div className="bg-orange-50 p-5 rounded-lg border border-orange-200">
            <h4 className="font-semibold text-orange-900 mb-3">⚠️ Common Pitfalls</h4>
            <ul className="space-y-2 text-sm text-gray-700">
              {data.bestPractices.donts.map((item, idx) => (
                <li key={idx}>• {item}</li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {/* Advanced Techniques */}
      {data.advancedTechniques && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-800">Advanced Techniques</h3>
          {data.advancedTechniques.map((technique, idx) => (
            <div key={idx} className="bg-purple-50 p-5 rounded-lg border border-purple-200">
              <h4 className="font-semibold text-purple-900 mb-2">{idx + 1}. {technique.title}</h4>
              <p className="text-sm text-gray-700 mb-3">{technique.description}</p>
              {technique.code && (
                <div className="bg-white p-3 rounded text-xs">
                  <code>{technique.code}</code>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Performance Optimization */}
      {data.performance && (
        <div className="bg-indigo-50 p-6 rounded-lg border border-indigo-200">
          <h3 className="text-lg font-semibold text-indigo-900 mb-3">🚀 {data.performance.title}</h3>
          <ul className="space-y-2 text-sm text-gray-700">
            {data.performance.tips.map((tip, idx) => (
              <li key={idx}>{tip}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Debugging */}
      {data.debugging && (
        <div className="bg-red-50 p-6 rounded-lg border border-red-200">
          <h3 className="text-lg font-semibold text-red-900 mb-3">🔧 {data.debugging.title}</h3>
          <div className="space-y-3 text-sm">
            {data.debugging.issues.map((issue, idx) => (
              <div key={idx}>
                <p className="font-semibold text-red-800">{issue.problem}</p>
                <p className="text-gray-700 ml-4">→ {issue.solution}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};