'use client';

import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import dynamic from 'next/dynamic';
import * as turf from '@turf/turf';
import api from '@/lib/api'; // Use the axios instance with Auth interceptor
import { useAuthStore } from '@/lib/store';
import { Sliders, X, Save, Key, Loader2, LogOut, StopCircle, Languages, Wrench, Search, Map, Calendar, Settings, ChevronDown, ChevronRight, List, Info, FileText, Sparkles } from 'lucide-react';
import ScraperDebugger from './ScraperDebugger';
import ReactMarkdown from 'react-markdown';
import { CAPITALS } from '../utils/capitals';
import DigestReportRenderer from './DigestReportRenderer';
import UIMarquee from './UIMarquee';

// Dynamically import Globe to avoid SSR issues
const Globe = dynamic(() => import('react-globe.gl'), {
    ssr: false,
    loading: () => <div className="flex items-center justify-center h-full text-white">Loading Clean Globe...</div>
});

interface NewsGlobeProps {
    onCountrySelect?: (countryName: string, countryCode: string) => void;
}

const isDateCurrent = (extractedDate: string) => {
    if (!extractedDate) return false;
    try {
        const d = new Date(extractedDate);
        if (isNaN(d.getTime())) return false;
        const now = new Date();
        const diff = now.getTime() - d.getTime();
        // Fresh if within 72 hours (3 days)
        return diff < 72 * 60 * 60 * 1000;
    } catch { return false; }
};

export default function NewsGlobe({ onCountrySelect }: NewsGlobeProps) {
    const globeEl = useRef<any>(null);

    // Initial View: Center on Romania (Lat ~46, Lng ~25)
    useEffect(() => {
        if (globeEl.current) {
            // Slight delay to ensure globe is ready
            setTimeout(() => {
                globeEl.current.pointOfView({ lat: 45.9432, lng: 24.9668, altitude: 2.0 }, 2000);
            }, 500);
        }
    }, []);
    const [countries, setCountries] = useState({ features: [] });
    const [selectedCountry, setSelectedCountry] = useState<any | null>(null);
    const [cities, setCities] = useState<any[]>([]);
    const [hoverPoint, setHoverPoint] = useState<any | null>(null);
    const [digestData, setDigestData] = useState<any>(null);
    const [isTranslateActive, setIsTranslateActive] = useState(false);
    const [debuggerConfig, setDebuggerConfig] = useState<{ isOpen: boolean; url: string; domain: string }>({
        isOpen: false,
        url: '',
        domain: ''
    });

    // Discovery Features
    const [discoveredCities, setDiscoveredCities] = useState<string[]>([]);
    const [allOutlets, setAllOutlets] = useState<any[]>([]);
    const [selectedCityOutlets, setSelectedCityOutlets] = useState<any[]>([]);
    const [selectedCityName, setSelectedCityName] = useState<string | null>(null);
    const [selectedCityData, setSelectedCityData] = useState<any>(null);
    const [selectedCityCoords, setSelectedCityCoords] = useState<{ lat: number; lng: number }>({ lat: 46.7712, lng: 23.6236 });

    interface Outlet {
        id?: number;
        name: string;
        url: string;
        type: string;
        origin?: string;
        popularity?: number; // 1-10
        focus?: string; // Local, National
    }

    interface AnalyticsKeyword {
        word: string;
        translation?: string;
        importance: number;
        sentiment: 'Positive' | 'Negative' | 'Neutral';
        source_ids: string[];
    }

    // UI States
    const [isDiscovering, setIsDiscovering] = useState(false);
    const [showOutletPanel, setShowOutletPanel] = useState(false);
    const [showAddForm, setShowAddForm] = useState(false);
    const [activeTab, setActiveTab] = useState<'manual' | 'import'>('import');
    const [newOutlet, setNewOutlet] = useState({ name: '', url: '', type: 'Online' });
    const [importUrl, setImportUrl] = useState('');
    const [importInstructions, setImportInstructions] = useState('');
    const [isImporting, setIsImporting] = useState(false);
    const [errorMessage, setErrorMessage] = useState<string | null>(null);
    const [quotaError, setQuotaError] = useState(false);

    // Search State
    const [searchQuery, setSearchQuery] = useState('');
    const [searchResults, setSearchResults] = useState<any[]>([]);
    const [highlightedCityId, setHighlightedCityId] = useState<string | null>(null); // For Halo effect

    // Aggregation Agent State
    const [selectedCategory, setSelectedCategory] = useState<string>('Politics');
    const [selectedTimeframe, setSelectedTimeframe] = useState<string>('24h');
    const [analyticsMode, setAnalyticsMode] = useState<'source' | 'digest'>('source');
    const [activeTooltip, setActiveTooltip] = useState<{ word: string, align: 'left' | 'right' | 'center', placement: 'top' | 'bottom' } | null>(null);
    const hoverTimeout = useRef<NodeJS.Timeout | null>(null);
    const [selectedOutletIds, setSelectedOutletIds] = useState<number[]>([]);
    const [isGeneratingDigest, setIsGeneratingDigest] = useState(false);
    const [cityInfo, setCityInfo] = useState<any>(null);

    // Logging State for Digest Generation
    const [progressLog, setProgressLog] = useState('');

    const [isSaving, setIsSaving] = useState(false);
    const [savedDigests, setSavedDigests] = useState<any[]>([]);
    const [activeSideTab, setActiveSideTab] = useState<'sources' | 'digests'>('sources');
    const [activeModalTab, setActiveModalTab] = useState<'articles' | 'digest' | 'analytics'>('articles');

    // Digest Summarization State
    const [selectedArticleUrls, setSelectedArticleUrls] = useState<Set<string>>(new Set());
    const [digestSummary, setDigestSummary] = useState<string>("");
    const [isSummarizing, setIsSummarizing] = useState(false);

    // Analytics State
    const [analyticsKeywords, setAnalyticsKeywords] = useState<AnalyticsKeyword[]>([]);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [isAnalyticsTranslated, setIsAnalyticsTranslated] = useState(false);
    const [analyticsViewMode, setAnalyticsViewMode] = useState<'cloud' | 'columns'>('cloud');
    const [analyzingTickerText, setAnalyzingTickerText] = useState<string>('');

    const handleToggleSelection = (url: string) => {
        const newSet = new Set(selectedArticleUrls);
        if (newSet.has(url)) newSet.delete(url);
        else newSet.add(url);
        setSelectedArticleUrls(newSet);
    };

    // Stable list of selected articles for Index-based citation
    const selectedArticlesList = useMemo(() => {
        if (!digestData?.articles) return [];
        return digestData.articles.filter((a: any) => selectedArticleUrls.has(a.url));
    }, [digestData, selectedArticleUrls]);

    // Ticker Effect
    useEffect(() => {
        if (!isSummarizing || selectedArticlesList.length === 0) {
            setAnalyzingTickerText('');
            return;
        }
        let idx = 0;
        const interval = setInterval(() => {
            const art = selectedArticlesList[idx % selectedArticlesList.length];
            const title = art.title || "Article";
            setAnalyzingTickerText(`Analyzing: ${title.substring(0, 40)}...`);
            idx++;
        }, 150);
        return () => clearInterval(interval);
    }, [isSummarizing, selectedArticlesList]);

    const handleSmartSelect = () => {
        if (!digestData?.articles) return;
        const smartSet = new Set<string>();
        digestData.articles.forEach((a: any) => {
            const s = a.scores || {};
            const isGreenDate = s.is_fresh || (a.relevance_score > 0 && s.date > 0);
            // "Green AI-Title" check
            const isVerified = a.ai_verdict === "VERIFIED";

            if (isGreenDate && isVerified) {
                smartSet.add(a.url);
            }
        });
        setSelectedArticleUrls(smartSet);
    };

    const handleGenerateBackendSummary = async () => {
        if (selectedArticleUrls.size === 0) return;
        setIsSummarizing(true);
        // "Gathering Articles..." text update
        // We handle the text in the button, but if there's a global loader, we can set it here.
        // For now, the user asked for text "somewhere". We updated the button text logic below.
        try {
            const selectedArts = digestData.articles.filter((a: any) => selectedArticleUrls.has(a.url));

            const res = await api.post('/outlets/digest/summarize', {
                articles: selectedArts,
                category: selectedCategory,
                city: selectedCityName || "Global"
            });

            setDigestSummary(res.data.summary);
            setActiveModalTab('digest');
        } catch (e) {
            console.error("Summarization failed", e);
            alert("Failed to generate summary");
        } finally {
            setIsSummarizing(false);
        }
    };

    const handleGenerateAnalytics = async () => {
        setIsAnalyzing(true);
        try {
            // Analyze selected articles if any, otherwise all available articles
            let targetArticles = digestData.articles;
            if (selectedArticleUrls.size > 0) {
                targetArticles = digestData.articles.filter((a: any) => selectedArticleUrls.has(a.url));
            }

            if (!targetArticles || targetArticles.length === 0) {
                alert("No articles available to analyze.");
                return;
            }

            const res = await api.post('/outlets/digest/analytics', {
                articles: targetArticles,
                category: selectedCategory,
                city: selectedCityName || "Global"
            });
            setAnalyticsKeywords(res.data.keywords);
        } catch (e) {
            console.error("Analytics failed", e);
            alert("Failed to generate analytics");
        } finally {
            setIsAnalyzing(false);
        }
    };

    const { isAuthenticated } = useAuthStore();

    useEffect(() => {
        if (activeSideTab === 'digests' && isAuthenticated) {
            fetchSavedDigests();
        }
    }, [activeSideTab, isAuthenticated]);

    const fetchSavedDigests = async () => {
        try {
            const res = await api.get('/digests');
            setSavedDigests(res.data);
        } catch (err) {
            console.error("Failed to load digests", err);
        }
    };
    // Scraper Debugger State
    const [scraperDebuggerOpen, setScraperDebuggerOpen] = useState(false);
    const [debugTarget, setDebugTarget] = useState<{ url: string, domain: string } | null>(null);

    const handleOpenDebugger = (url: string) => {
        try {
            const domain = new URL(url).hostname.replace('www.', '');
            setDebugTarget({ url, domain });
            setScraperDebuggerOpen(true);
        } catch (e) {
            console.error("Invalid URL for debugger", url);
        }
    };




    const handleRuleSaving = () => {
        if (!debugTarget || !digestData) return;

        let newDigestHtml = digestData.digest;
        if (typeof document !== 'undefined') {
            try {
                const tempDiv = document.createElement('div');
                tempDiv.innerHTML = newDigestHtml;

                const triggers = tempDiv.querySelectorAll(`.scraper-debug-trigger[data-url="${debugTarget.url}"]`);
                triggers.forEach((trigger: any) => {
                    if (trigger.previousElementSibling) {
                        trigger.previousElementSibling.innerHTML = "⏳";
                        trigger.previousElementSibling.className = "animate-spin inline-block";
                    }
                });

                newDigestHtml = tempDiv.innerHTML;
            } catch (e) { console.warn("Saving spinner patch failed", e); }
        }

        setDigestData({
            ...digestData,
            digest: newDigestHtml
        });
    }

    // Live Rule Update Handler
    const handleRuleSaved = async (domain: string) => {
        if (!digestData || !digestData.articles) return;

        const relevantArticles = digestData.articles.filter((art: any) => {
            try {
                return new URL(art.url).hostname.replace('www.', '') === domain;
            } catch (e) { return false; }
        });

        if (relevantArticles.length === 0) return;

        const updatedArticles = [...digestData.articles];
        let newDigestHtml = digestData.digest;

        for (const art of relevantArticles) {
            try {
                // Apply Rule Test
                const response = await api.post('/scraper/test', { url: art.url });

                if (response.data.status === 'success' && response.data.extracted_date) {
                    const idx = updatedArticles.findIndex(a => a.url === art.url);
                    if (idx !== -1) {
                        const newDate = response.data.extracted_date;
                        const isFresh = isDateCurrent(newDate);

                        updatedArticles[idx] = {
                            ...updatedArticles[idx],
                            date_str: newDate,
                            scores: {
                                ...updatedArticles[idx].scores,
                                is_fresh: isFresh,
                                is_old: !isFresh
                            },
                            relevance_score: isFresh ? updatedArticles[idx].relevance_score : 0
                        };

                        // HTML Patch
                        if (typeof document !== 'undefined') {
                            try {
                                const tempDiv = document.createElement('div');
                                tempDiv.innerHTML = newDigestHtml;
                                const trigger = tempDiv.querySelector(`.scraper-debug-trigger[data-url="${CSS.escape(art.url)}"]`);

                                if (trigger && trigger.previousElementSibling) {
                                    trigger.previousElementSibling.textContent = newDate;
                                    trigger.previousElementSibling.className = ""; // remove spin
                                    // Color logic
                                    const dateEl = trigger.previousElementSibling as HTMLElement;
                                    dateEl.style.color = isFresh ? "#4ade80" : "#7f1d1d";

                                    // PROMOTION LOGIC: Move to top table if fresh
                                    if (isFresh) {
                                        const row = trigger.closest('tr');
                                        if (row) {
                                            const currentTable = row.closest('table');
                                            // Assume first table is "Relevant"/Fresh
                                            const mainTable = tempDiv.querySelector('table');

                                            if (currentTable && mainTable && currentTable !== mainTable) {
                                                // It's in a secondary table (Older?), move it!
                                                const tbody = mainTable.querySelector('tbody') || mainTable;
                                                // Insert at top or bottom? User said "promoted", maybe bottom of fresh is fine.
                                                // Or top? Let's append for now.
                                                tbody.appendChild(row);

                                                // Optional: Add a highlight effect class effectively?
                                                // Can't easily animate via string injection, but the move is enough.
                                            }
                                        }
                                    }
                                }
                                newDigestHtml = tempDiv.innerHTML;
                            } catch (e) { console.warn("HTML Patch failed", e); }
                        }
                    }
                }
            } catch (e) {
                console.warn("Failed to re-scrape", art.url, e);
            }
        }

        setDigestData({
            ...digestData,
            articles: updatedArticles,
            digest: newDigestHtml
        });
    };

    // Actually, backend sets SCORE to 0 if invalid. Let's use that signal for reliability.
    const isArticleValid = (art: any) => art.relevance_score > 0;

    const handleSaveDigest = async () => {
        if (!digestData) return;
        setIsSaving(true);
        try {
            // Extract Title from Markdown (First line # Title)
            const titleMatch = digestData.digest.match(/^# (.*)$/m);
            const title = titleMatch ? titleMatch[1] : `${selectedCategory} Digest`;

            await api.post('/digests', {
                title: title,
                category: selectedCategory,
                city: selectedCityName || "Global",
                timeframe: selectedTimeframe,
                summary_markdown: digestSummary || digestData.digest,
                articles: digestData.articles,
                selected_article_urls: Array.from(selectedArticleUrls),
                analysis_source: analyticsKeywords.length > 0 ? analyticsKeywords : digestData.analysis_source
            });
            await fetchSavedDigests();
            // Show toast? relying on button state for now
        } catch (err) {
            console.error("Failed to save digest", err);
            setErrorMessage("Failed to save digest.");
        } finally {
            setIsSaving(false);
        }
    };

    const handleDownloadDigest = () => {
        if (!digestData) return;

        const titleMatch = digestData.digest.match(/^# (.*)$/m);
        const title = titleMatch ? titleMatch[1] : `${selectedCategory} Digest`;
        const filename = `${title.replace(/[^a-z0-9]/gi, '_').toLowerCase()}.html`;

        const htmlContent = `
            <!DOCTYPE html>
            <html class="dark">
            <head>
                <meta charset="utf-8">
                <title>${title}</title>
                <script src="https://cdn.tailwindcss.com"></script>
                <style>
                    body { background-color: #020617; color: #e2e8f0; font-family: system-ui, sans-serif; padding: 2rem; max-width: 800px; margin: 0 auto; }
                    .prose { max-width: none; }
                    
                    /* Fallback Styles */
                    h1 { font-size: 2.25rem; font-weight: 800; margin-bottom: 1rem; color: #f8fafc; }
                    h2 { font-size: 1.5rem; font-weight: 700; margin-top: 2rem; margin-bottom: 1rem; color: #f1f5f9; border-bottom: 1px solid #1e293b; padding-bottom: 0.5rem; }
                    h3 { font-size: 1.25rem; font-weight: 600; margin-top: 1.5rem; margin-bottom: 0.75rem; color: #e2e8f0; }
                    a { color: #60a5fa; text-decoration: none; }
                    a:hover { text-decoration: underline; }
                    table { width: 100%; border-collapse: collapse; margin: 1.5rem 0; font-size: 0.875rem; }
                    th, td { border: 1px solid #1e293b; padding: 0.75rem; text-align: left; }
                    th { background-color: #0f172a; font-weight: 600; color: #94a3b8; }
                    tr:nth-child(even) { background-color: #0f172a; }
                    blockquote { border-left: 4px solid #3b82f6; padding-left: 1rem; color: #94a3b8; font-style: italic; }
                    ul { list-style-type: disc; padding-left: 1.5rem; margin: 1rem 0; }
                    
                    .politics-assess-trigger { display: none; } /* Hide interactive buttons */
                    .scraper-debug-trigger { pointer-events: none; opacity: 0.7; }
                </style>
            </head>
            <body>
                <article class="prose prose-invert">
                    ${digestData.digest}
                </article>
                <footer style="margin-top: 4rem; padding-top: 2rem; border-top: 1px solid #1e293b; color: #64748b; font-size: 0.875rem; text-align: center;">
                    Generated by OpenNews • ${new Date().toLocaleString()}
                </footer>
            </body>
            </html>
        `;

        const blob = new Blob([htmlContent], { type: 'text/html' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    };

    const abortControllerRef = useRef<AbortController | null>(null);

    const handleStopDigest = () => {
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
            abortControllerRef.current = null;
            setProgressLog("🛑 Stopped by user.");
            setIsGeneratingDigest(false);
        }
    };

    const handleGenerateDigest = async () => {
        if (!isAuthenticated) {
            setErrorMessage("Please log in to use AI features.");
            return;
        }

        if (selectedOutletIds.length === 0) {
            setErrorMessage("Please select at least one outlet.");
            return;
        }
        setIsGeneratingDigest(true);
        setErrorMessage(null);
        setDigestData(null);
        setSelectedArticleUrls(new Set());
        setDigestSummary("");
        setProgressLog('Connecting to stream...');

        try {
            // Retrieve token from storage/api helper if needed
            const token = localStorage.getItem('token');

            // Init AbortController
            abortControllerRef.current = new AbortController();

            const response = await fetch(`${api.defaults.baseURL}/outlets/digest/stream`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({
                    outlet_ids: selectedOutletIds,
                    category: selectedCategory,
                    timeframe: selectedTimeframe
                }),
                signal: abortControllerRef.current.signal
            });

            if (!response.body) throw new Error("No stream body");

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let done = false;
            let buffer = '';
            let lastLogUpdate = 0;
            let lastDataUpdate = 0;
            // Accumulator for throttled updates
            let currentDigestState: any = { articles: [], digest: '', analysis_source: [] };

            while (!done) {
                const { value, done: doneReading } = await reader.read();
                done = doneReading;
                const chunkValue = decoder.decode(value || new Uint8Array(), { stream: !done });

                buffer += chunkValue;
                const lines = buffer.split('\n');

                // Keep the last line in the buffer as it might be incomplete
                // unless we are done, in which case process everything.
                buffer = done ? '' : lines.pop() || '';

                for (const line of lines) {
                    if (line.trim() === '') continue;
                    try {
                        const msg = JSON.parse(line);

                        if (msg.type === 'log') {
                            // Throttle log updates to prevent UI/WebGL thrashing (max 5fps)
                            const now = Date.now();
                            if (now - lastLogUpdate > 200 || msg.message.includes("Done") || msg.message.includes("Error")) {
                                setProgressLog(`> ${msg.message}`);
                                lastLogUpdate = now;
                            }
                        }
                        // --- New Partial Handlers ---
                        else if (msg.type === 'partial_digest') {
                            // console.log("Received Partial Digest HTML");
                            currentDigestState.digest = msg.html;

                            const now = Date.now();
                            if (now - lastDataUpdate > 200) {
                                setDigestData({ ...currentDigestState });
                                lastDataUpdate = now;
                            }
                        }
                        else if (msg.type === 'partial_articles') {
                            // console.log(`Received Partial Articles List (${msg.articles.length})`);
                            if (currentDigestState.articles) {
                                currentDigestState.articles.push(...msg.articles);
                            } else {
                                currentDigestState.articles = [...msg.articles];
                            }
                            currentDigestState.category = msg.category;

                            const now = Date.now();
                            if (now - lastDataUpdate > 200) {
                                setDigestData({ ...currentDigestState });
                                lastDataUpdate = now;
                            }
                        }
                        else if (msg.type === 'partial_analysis') {
                            // console.log("Received Partial Analysis");
                            currentDigestState.analysis_source = msg.source;

                            const now = Date.now();
                            if (now - lastDataUpdate > 200) {
                                setDigestData({ ...currentDigestState });
                                lastDataUpdate = now;
                            }
                        }
                        else if (msg.type === 'done') {
                            console.log("Digest Stream Complete.");
                            // Final Update
                            setDigestData({ ...currentDigestState });

                            // Auto-Select Fresh & Verified
                            const autoSelected = new Set<string>(
                                currentDigestState.articles
                                    .filter((a: any) => {
                                        const s = a.scores || {};
                                        const isFresh = s.is_fresh || (a.relevance_score > 0 && s.date > 0);
                                        const isVerified = a.ai_verdict === "VERIFIED";
                                        return isFresh && isVerified;
                                    })
                                    .map((a: any) => a.url)
                            );
                            setSelectedArticleUrls(autoSelected);

                            setActiveModalTab('articles');
                            setShowOutletPanel(true);

                            // Check for rate limits in the accumulated data
                            if (currentDigestState.analysis_source) {
                                const hasRateLimit = currentDigestState.analysis_source.some((k: any) => k.type === "System:RateLimit");
                                if (hasRateLimit) {
                                    alert("⚠️ Rate Limit Warning: Some articles could not be fully analyzed.");
                                }
                            }
                        }
                        // --- Legacy Fallback ---
                        else if (msg.type === 'result') {
                            const data = msg.payload;
                            console.log("Received Digest Result:", data); // Debug
                            currentDigestState = data; // Update local
                            setDigestData(data);

                            // Auto-Select Fresh & Verified
                            const autoSelectedResult = new Set<string>(
                                data.articles
                                    .filter((a: any) => {
                                        const s = a.scores || {};
                                        const isFresh = s.is_fresh || (a.relevance_score > 0 && s.date > 0);
                                        const isVerified = a.ai_verdict === "VERIFIED";
                                        return isFresh && isVerified;
                                    })
                                    .map((a: any) => a.url)
                            );
                            setSelectedArticleUrls(autoSelectedResult);

                            setActiveModalTab('articles');
                            setShowOutletPanel(true);
                        } else if (msg.type === 'error') {
                            setErrorMessage(msg.message);
                        }
                    } catch (e) {
                        console.warn("Stream parse error for line:", line.substring(0, 50) + "...", e);
                    }
                }
            }

        } catch (err: any) {
            if (err.name === 'AbortError') {
                console.log("Digest generation aborted by user.");
                return; // Clean exit
            }
            console.error("Digest generation failed", err);
            if (err.message && err.message.includes("network")) {
                setErrorMessage("Network Timeout. Please try fewer outlets or a smaller timeframe.");
            } else {
                setErrorMessage(err.message || 'Failed to generate digest');
            }
        } finally {
            setIsGeneratingDigest(false);
        }
    };

    const handleLoadDigest = (digest: any) => {
        setDigestData({
            digest: digest.summary_markdown,
            articles: digest.articles,
            analysis_source: digest.analysis_source
        });
        setSelectedCategory(digest.category);
        setActiveModalTab('articles');
    };

    const handleDeleteDigest = async (e: any, id: number) => {
        e.stopPropagation();
        if (!confirm("Delete this digest?")) return;
        try {
            await api.delete(`/digests/${id}`);
            fetchSavedDigests();
        } catch (err) {
            console.error(err);
        }
    };

    const CATEGORIES = ['Politics', 'Internal Affairs', 'External Affairs', 'Sports', 'Business', 'Tech'];
    const countryMap = useRef<Record<string, string>>({});

    useEffect(() => {
        console.log("NewsGlobe Mounted. Fetching initial data...");
        // Load countries polygons
        fetch('https://raw.githubusercontent.com/vasturiano/react-globe.gl/master/example/datasets/ne_110m_admin_0_countries.geojson')
            .then(res => res.json())
            .then(data => {
                setCountries(data);
                const map: Record<string, string> = {};
                data.features.forEach((f: any) => {
                    if (f.properties && f.properties.ISO_A2) {
                        map[f.properties.ISO_A2] = f.properties.ADMIN;
                    }
                });
                countryMap.current = map;
            });

        // Load cities
        fetch('https://raw.githubusercontent.com/lmfmaier/cities-json/master/cities500.json')
            .then(res => res.json())
            .then(data => {
                const largeCities = Array.isArray(data)
                    ? data.filter((d: any) => parseInt(d.pop || 0) > 100000)
                    : [];
                setCities(largeCities);
            })
            .catch(err => console.error("Failed to load cities data", err));

        // Load initially discovered cities (Auth required)
        api.get('/outlets/cities/list')
            .then(res => {
                console.log("Discovered Cities Loaded:", res.data?.length);
                setDiscoveredCities(res.data);
            })
            .catch(err => console.error("Failed to load discovered cities", err));

        // Load all outlets for mapping
        api.get('/outlets/')
            .then(res => {
                console.log("All Outlets Loaded:", res.data?.length);
                setAllOutlets(res.data);
            })
            .catch(err => console.error("Failed to load outlets map", err));
    }, []);

    // Search Logic
    useEffect(() => {
        if (!searchQuery || searchQuery.length < 2) {
            setSearchResults([]);
            return;
        }

        // Normalize helper: remove diacritics (e.g. ș -> s, ă -> a)
        const normalizeText = (text: string) =>
            text.normalize('NFD').replace(/[\u0300-\u036f]/g, "").toLowerCase();

        const lowerQuery = normalizeText(searchQuery);

        // Search in processedData (to find active nodes) or raw cities?
        // Raw cities is better for complete coverage.
        const results = cities.filter(c =>
            c.name && normalizeText(c.name).includes(lowerQuery)
        ).slice(0, 10);

        setSearchResults(results);
    }, [searchQuery, cities]);

    const handleDebugArticle = (article: any) => {
        let domain = article.source;
        // CRITICAL: Backend matches rules by DOMAIN (e.g. "g4media.ro"), not Outlet Name (e.g. "G4Media").
        // We must extract the actual hostname from the URL for the rule to be effective.
        try {
            if (article.url) {
                const urlObj = new URL(article.url);
                domain = urlObj.hostname.replace('www.', '');
            }
        } catch (e) {
            console.error("Failed to parse URL for debugger", e);
        }

        setDebuggerConfig({
            isOpen: true,
            url: article.url,
            domain: domain
        });
    };

    const handleSearchSelect = (city: any) => {
        setSearchQuery('');
        setSearchResults([]);

        // Find the node in the globe data
        // It's either a point, or inside a cluster point.
        let targetNode = processedData.points.find(p => p.id === city.id || p.name === city.name);
        let parentCluster = null;

        if (!targetNode) {
            // Check inside clusters
            for (const p of processedData.points) {
                if (p.isCluster && p.subPoints) {
                    const found = p.subPoints.find((sub: any) => sub.name === city.name);
                    if (found) {
                        targetNode = found; // The city itself
                        parentCluster = p; // The cluster it belongs to
                        break;
                    }
                }
            }
        }

        // Coordinate safety check
        // Raw dataset often uses 'lon', processed might use 'lng'
        const rawLat = city.lat || city.latitude;
        const rawLng = city.lng || city.lon || city.longitude;
        const targetLat = parseFloat(rawLat);
        const targetLng = parseFloat(rawLng);

        if (isNaN(targetLat) || isNaN(targetLng)) {
            console.error("Invalid coordinates for city:", city);
            return;
        }

        if (targetNode || parentCluster) {
            // If in cluster, Expand Cluster first
            if (parentCluster) {
                setExpandedCluster(parentCluster);
                // The view will re-render, showing the spiders.
                if (globeEl.current) {
                    globeEl.current.pointOfView({ lat: targetLat, lng: targetLng, altitude: 0.1 }, 1500);
                }
            } else {
                if (globeEl.current) {
                    globeEl.current.pointOfView({ lat: targetLat, lng: targetLng, altitude: 0.1 }, 1500);
                }
            }

            // Set Halo
            setHighlightedCityId(city.name); // Using Name as ID for highlight matching
            setTimeout(() => setHighlightedCityId(null), 4000); // Remove halo after 4s

            // Select it (Trigger sidebar)
            // handleCityClick requires the 'd' object. 
            // If we found a targetNode (from processed data), use that as it has correct structure/stats.
            // If not (searched city not on map?), fallback to raw city but it might lack 'country' code used by handleCityClick.
            // But 'cities' dataset usually has country code.
            const cityToSelect = targetNode || { ...city, lat: targetLat, lng: targetLng, lon: targetLng };
            handleCityClick(cityToSelect);
        } else {
            // City found in search but not currently visualized (maybe filtered out?)
            // Just fly there and select it.
            if (globeEl.current) {
                globeEl.current.pointOfView({ lat: targetLat, lng: targetLng, altitude: 0.1 }, 1500);
            }
            // Set Halo
            setHighlightedCityId(city.name);
            setTimeout(() => setHighlightedCityId(null), 4000);

            const cityToSelect = { ...city, lat: targetLat, lng: targetLng, lon: targetLng };
            handleCityClick(cityToSelect);
        }
    };

    // --- Visual Controls ---
    const [clusterThreshold, setClusterThreshold] = useState(0.7); // Default 0.7 deg
    const [markerScale, setMarkerScale] = useState(0.6); // Default 0.6x
    const [showControls, setShowControls] = useState(false); // Toggle for Viz Controls

    const MAP_STYLES = [
        { name: 'Satellite', url: '//unpkg.com/three-globe/example/img/earth-blue-marble.jpg' },
        { name: 'Day', url: '//unpkg.com/three-globe/example/img/earth-day.jpg' },
        { name: 'Night', url: '//unpkg.com/three-globe/example/img/earth-night.jpg' },
        { name: 'Dark', url: '//unpkg.com/three-globe/example/img/earth-dark.jpg' }
    ];
    const [mapStyle, setMapStyle] = useState(MAP_STYLES[0].url);
    const [editingOutletId, setEditingOutletId] = useState<number | null>(null);
    const [editUrl, setEditUrl] = useState('');

    // --- Advanced Visualization State ---
    const [processedData, setProcessedData] = useState<{ points: any[], rings: any[], links: any[] }>({ points: [], rings: [], links: [] });
    const [expandedCluster, setExpandedCluster] = useState<any | null>(null);

    // Helpers
    const getPopScale = (pop: any) => {
        const val = parseInt(pop || '0');
        if (val < 1000) return 0.02 * markerScale;
        // Ensure visible minimum even with small scale
        // Log10(1M)=6 -> 0.18. * 0.6 = 0.108
        return Math.max(0.08, Math.log10(val) * 0.03) * markerScale;
    };



    // Clustering Logic (Simple Distance)
    // const CLUSTER_THRESHOLD = 2.5; // Degrees // Removed, using state variable

    const clusters = useMemo(() => {
        if (cities.length === 0) return [];

        // 0. Pre-calculate "Capitals"
        const countryMaxPop: Record<string, number> = {};
        cities.forEach(c => {
            const pop = parseInt(c.pop || '0');
            const country = c.country || 'XX';
            if (!countryMaxPop[country] || pop > countryMaxPop[country]) {
                countryMaxPop[country] = pop;
            }
        });

        // 1. Sort by pop desc
        const sorted = [...cities].sort((a, b) => parseInt(b.pop || 0) - parseInt(a.pop || 0));

        // 1.5 Calculate Active Cities from Digest (Live Mode)
        const activeDigestCities = new Set<string>();
        if (digestData?.articles && allOutlets.length > 0) {
            console.log(`[NewsGlobe] Mapping ${digestData.articles.length} articles against ${allOutlets.length} outlets.`);
            digestData.articles.forEach((art: any) => {
                try {
                    let outlet = null;
                    if (art.url) {
                        const artDomain = new URL(art.url).hostname.replace('www.', '');
                        outlet = allOutlets.find((o: any) => {
                            try { return new URL(o.url).hostname.replace('www.', '') === artDomain; } catch { return false; }
                        });
                    }
                    if (!outlet && art.source) {
                        // Loose match on source name
                        outlet = allOutlets.find((o: any) => o.name === art.source);
                    }
                    if (outlet && outlet.city) {
                        activeDigestCities.add(outlet.city);
                    }
                } catch (e) { }
            });
            console.log(`[NewsGlobe] Found ${activeDigestCities.size} active cities.`);
        }

        const newClusters: any[] = [];

        sorted.forEach(city => {
            const lat = parseFloat(city.lat);
            const lng = parseFloat(city.lon);
            const pop = parseInt(city.pop || '0');

            city.radius = getPopScale(city.pop);
            city.isCluster = false;

            const isDiscovered = discoveredCities.includes(city.name);

            // const isCapital = pop === countryMaxPop[city.country || 'XX']; // Old heuristic
            // Exact match check. Note: city.country is ISO2, same as CAPITALS keys.
            const isCapital = CAPITALS[city.country] === city.name;

            let color = isDiscovered ? '#34d399' : (isCapital ? '#db2777' : '#64748b');

            // LIVE OVERRIDE
            if (activeDigestCities.has(city.name)) {
                color = '#4ade80'; // Bright Green for Active News
            }

            city.color = color;
            city.isCapital = isCapital;

            // Find existing cluster
            const existing = newClusters.find(c => {
                if (c.country !== city.country) return false;
                const dist = Math.sqrt(Math.pow(c.lat - lat, 2) + Math.pow(c.lng - lng, 2));
                return dist < clusterThreshold;
            });

            if (existing) {
                // If the new city is a Capital and the Cluster is NOT yet a Capital,
                // we SWAP them. The Capital becomes the Head of the cluster.
                // The old Head (e.g. Johannesburg) becomes a sub-point.
                if (city.isCapital && !existing.isCapital) {
                    // 1. Save Old Head as a Subpoint
                    const oldHead = {
                        ...existing,
                        // Important: Restore original identity data we might want
                        // But wait, 'existing' is the accumulator object. 
                        // We need to construct a "Point" object for the Old Head.
                        // We rely on 'ownPop' which we must add to initialization.
                        pop: existing.ownPop,
                        subPoints: [], // Subpoint is a leaf
                        isCluster: false,
                        radius: getPopScale(existing.ownPop),
                        color: discoveredCities.includes(existing.name) ? '#34d399' : '#64748b'
                        // Note: Re-calculating color is safer. 
                        // We don't have 'isDiscovered' on existing easily unless we updated it. 
                        // But existing.color is currently set. Let's use it? 
                        // existing.color might be purple (Cluster color).
                        // Better to use default gray usually, or check discovered.
                        // Let's assume generic gray for simplicity or re-derive if needed.
                        // Actually, let's just make it a standard point.
                    };
                    delete oldHead.subPoints; // Ensure it's clean
                    delete oldHead.points; // logic safety

                    existing.subPoints.push(oldHead);

                    // 2. Update Head Identity to New Capital
                    existing.name = city.name;
                    existing.lat = lat;
                    existing.lng = lng;
                    existing.lon = lng;
                    existing.country = city.country;
                    existing.ownPop = pop;
                    existing.isCapital = true;
                    existing.color = '#db2777'; // Capital Red

                    // 3. Update Totals
                    existing.pop += pop; // Add Capital Pop to Total
                    existing.radius = getPopScale(existing.pop) * 1.2;

                } else {
                    // Standard Merge: New city becomes a subpoint
                    existing.subPoints.push(city);
                    existing.pop += pop;
                    existing.radius = getPopScale(existing.pop) * 1.2;

                    if (existing.isCapital) {
                        // Head is already capital, maintain color
                    } else if (city.isCapital) {
                        // Edge case: Multiple capitals? 
                        // Should not happen if we did the swap above (since !existing.isCapital).
                        // But if existing IS capital AND city IS capital (2 capitals close?),
                        // We just swallow the smaller capital.
                    } else {
                        // No capitals involved
                        existing.color = '#7c3aed';
                    }
                }
                existing.isCluster = true;

            } else {
                newClusters.push({
                    ...city,
                    lat: lat,
                    lng: lng,
                    lon: lng,
                    subPoints: [],
                    id: `c-${lat}-${lng}`,
                    pop: pop,
                    ownPop: pop, // Save personal population for swapping
                    isCapital: isCapital
                });
            }
        });
        return newClusters;
    }, [cities, discoveredCities, clusterThreshold, markerScale, digestData, allOutlets]); // Updated deps

    // 2. Generate Render Objects based on Expanded State (Fast)
    useEffect(() => {
        const renderPoints: any[] = [];
        const renderRings: any[] = [];
        const renderLinks: any[] = [];
        const isAnyExpanded = !!expandedCluster;

        clusters.forEach((c: any) => {
            if (expandedCluster) {
                // FOCUS MODE: Only render the expanded cluster logic.
                // Skip everything else.
                if (c.id === expandedCluster.id) {
                    const items = [c, ...c.subPoints];
                    const count = items.length;
                    const goldenAngle = Math.PI * (3 - Math.sqrt(5));
                    const spreadFactor = clusterThreshold * 0.5;

                    // 1. Prepare Items with Parsed Coordinates
                    const preparedItems = items.map((item: any) => ({
                        ...item,
                        pLat: parseFloat(item.lat || item.latitude),
                        pLng: parseFloat(item.lng || item.lon || item.longitude),
                        pRadius: getPopScale(item.pop)
                    }));

                    // 2. Iterative Relaxation (Force-Directed Packing)
                    // Instead of a single global scalar, we simulate "pushing" overlapping circles apart.

                    const ITERATIONS = 50; // increased for dense clusters (e.g. Tokyo)
                    const padding = 0.035; // increased clearance
                    console.log("Using Force Packing for Cluster:", c.name);

                    // Clone for simulation
                    const simItems = preparedItems.map((p: any) => ({
                        ...p,
                        x: p.pLng, // Longitude ~ X
                        y: p.pLat, // Latitude ~ Y
                        r: p.pRadius,
                        vx: 0,
                        vy: 0
                    }));

                    for (let iter = 0; iter < ITERATIONS; iter++) {
                        let moved = false;
                        for (let i = 0; i < simItems.length; i++) {
                            for (let j = i + 1; j < simItems.length; j++) {
                                const p1 = simItems[i];
                                const p2 = simItems[j];

                                const dx = p2.x - p1.x;
                                const dy = p2.y - p1.y;
                                const distSq = dx * dx + dy * dy;
                                const dist = Math.sqrt(distSq);

                                const minDist = p1.r + p2.r + padding;

                                if (dist < minDist) {
                                    // Overlap!
                                    const overlap = minDist - dist;
                                    const nx = dist > 0 ? dx / dist : 1; // Normalize vector
                                    const ny = dist > 0 ? dy / dist : 0;

                                    // Push apart
                                    const moveX = nx * (overlap * 0.51);
                                    const moveY = ny * (overlap * 0.51);

                                    if (i !== 0) {
                                        p1.x -= moveX;
                                        p1.y -= moveY;
                                    }

                                    if (j !== 0) {
                                        p2.x += moveX;
                                        p2.y += moveY;
                                    }

                                    moved = true;
                                }
                            }
                        }
                        if (!moved) break;
                    }

                    simItems.forEach((item: any, idx: number) => {
                        const isCenter = idx === 0;
                        const exLat = item.y;
                        const exLng = item.x;

                        let itemColor = '#a78bfa';
                        if (discoveredCities.includes(item.name)) itemColor = '#34d399';
                        else if (item.isCapital) itemColor = '#db2777';

                        renderPoints.push({
                            ...item, // include original props
                            lat: exLat,
                            lng: exLng,
                            lon: exLng,
                            color: itemColor,
                            radius: item.pRadius, // use parsed radius
                            opacity: 1.0,
                            isSpider: true
                        });

                        if (!isCenter) {
                            renderLinks.push({
                                startLat: c.lat,
                                startLng: c.lng,
                                endLat: exLat,
                                endLng: exLng,
                                color: 'rgba(255,255,255,0.3)'
                            });
                        }
                    });
                    // Adjust ring to cover the area?
                    renderRings.push({ lat: c.lat, lng: c.lng, maxR: spreadFactor * 2.0, color: 'rgba(255,255,255,0.05)' });
                }
                // ELSE: Do nothing. Hide node.
            } else {
                // Determine opacity based on... actually, if nothing expanded, full opacity.
                renderPoints.push({ ...c, opacity: 1.0 });

                if (c.isCluster) {
                    renderRings.push({
                        lat: c.lat,
                        lng: c.lng,
                        maxR: c.radius * 1.5,
                        color: 'rgba(124, 58, 237, 0.3)'
                    });
                }
            }
        });

        setProcessedData({ points: renderPoints, rings: renderRings, links: renderLinks });
    }, [clusters, expandedCluster]);


    // ESC Key to close cluster
    useEffect(() => {
        const handleEsc = (e: KeyboardEvent) => {
            if (e.key === 'Escape') setExpandedCluster(null);
        };
        window.addEventListener('keydown', handleEsc);
        return () => window.removeEventListener('keydown', handleEsc);
    }, []);

    const getTooltip = useCallback((d: any) => {
        if (d.isCluster && !d.isSpider) {
            return `
            <div class="px-2 py-1 bg-amber-500/90 text-black font-bold rounded text-xs border border-amber-300 z-50">
                <div class="text-[10px] uppercase opacity-80">Media Hub</div>
                <div class="text-sm">${d.name}</div>
                <div class="text-[10px]">+ ${d.subPoints.length} cities</div>
            </div>
            `;
        }
        return `
            <div style="background: rgba(0,0,0,0.9); color: ${d.color}; padding: 4px 8px; border-radius: 4px; font-family: sans-serif; font-weight: bold; font-size: 12px; border: 1px solid ${d.color}; pointer-events: none;">
                ${d.name} <span style="opacity:0.7">(${parseInt(d.pop || 0).toLocaleString()})</span>
            </div>
            `;
    }, []);

    const handleCityClick = useCallback((d: any) => {
        setSelectedCityName(d.name);
        setSelectedCityData(d);
        setShowOutletPanel(true);
        setIsDiscovering(true);
        setSelectedCityOutlets([]);
        setShowAddForm(false);
        setActiveTab('import');
        setImportUrl('');
        setImportInstructions('');
        setErrorMessage(null);
        setDigestData(null);

        const countryCode = d.country || "XX";
        const countryName = countryMap.current[countryCode] || countryCode;

        api.post('/outlets/discover_city', {
            city: d.name,
            country: countryName,
            lat: parseFloat(d.lat),
            lng: parseFloat(d.lon)
        })
            .then(res => {
                const data = res.data;
                if (Array.isArray(data)) {
                    setSelectedCityOutlets(data);
                    setSelectedOutletIds(data.map((o: any) => o.id));
                    if (data.length > 0 && !discoveredCities.includes(d.name)) {
                        setDiscoveredCities(prev => [...prev, d.name]);
                    }
                }
            })
            .catch(err => {
                if (err.response?.status === 429) {
                    setQuotaError(true);
                    setErrorMessage("Quota Exceeded");
                } else {
                    console.error("Discovery failed", err);
                }
            })
            .finally(() => setIsDiscovering(false));

        // Get City Info
        setCityInfo(null);
        api.get(`/outlets/city_info?city=${d.name}&country=${countryName}`)
            .then(res => setCityInfo(res.data))
            .catch(err => console.error("City Info failed", err));
    }, [discoveredCities]);

    const handleMapClick = useCallback((d: any) => {
        // If Spider Point (previously grouped), treat as city click
        if (d.isSpider || !d.isCluster) {
            handleCityClick(d);
            return;
        }

        // If Cluster, Expand
        if (d.isCluster) {
            if (expandedCluster && expandedCluster.id === d.id) {
                setExpandedCluster(null); // Collapse
                // Reset zoom? optional
            } else {
                const center = turf.point([parseFloat(d.lon), parseFloat(d.lat)]);
                // Adaptive Zoom: Closer! (User Req: "more zoom in")
                const count = d.subPoints.length + 1;
                // Min altitude 0.025 for extremely close view
                const adaptiveAlt = Math.max(0.025, 0.25 - (count * 0.008));

                if (globeEl.current) {
                    globeEl.current.pointOfView({ lat: d.lat, lng: d.lng, altitude: adaptiveAlt }, 800);
                }
                setExpandedCluster(d);
            }
        }
    }, [expandedCluster, handleCityClick]);


    const handleRediscoverCity = () => {
        if (!selectedCityData) return;
        setIsDiscovering(true);
        setQuotaError(false);
        setSelectedCityOutlets([]);

        const d = selectedCityData;
        const countryCode = d.country || "XX";
        const countryName = countryMap.current[countryCode] || countryCode;

        api.post('/outlets/discover_city', {
            city: d.name,
            country: countryName,
            lat: parseFloat(d.lat),
            lng: parseFloat(d.lon),
            force_refresh: true
        })
            .then(res => {
                const data = res.data;
                if (Array.isArray(data)) {
                    setSelectedCityOutlets(data);
                    setSelectedOutletIds(data.map((o: any) => o.id));
                }
            })
            .catch(err => {
                if (err.response?.status === 429) setQuotaError(true);
            })
            .finally(() => setIsDiscovering(false));
    };

    const toggleOutletSelection = (id: number) => {
        setSelectedOutletIds(prev =>
            prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]
        );
    };



    const handleDeleteOutlet = (e: any, outletId: number) => {
        e.stopPropagation();
        if (!confirm('Are you sure you want to delete this outlet?')) return;
        api.delete(`/outlets/${outletId}`)
            .then(() => {
                setSelectedCityOutlets(prev => prev.filter(o => o.id !== outletId));
            })
            .catch(err => alert(`Failed to delete: ${err.message}`));
    };

    const handleUpdateOutlet = (outletId: number) => {
        if (!editUrl) return;
        api.put(`/outlets/${outletId}`, { url: editUrl })
            .then(res => {
                setSelectedCityOutlets(prev => prev.map(o => o.id === outletId ? { ...o, url: res.data.url } : o));
                setEditingOutletId(null);
            })
            .catch(err => alert(`Failed to update: ${err.message}`));
    };

    const handleImportUrl = (e: React.FormEvent) => {
        e.preventDefault();
        if (!selectedCityData || !importUrl) return;
        setIsImporting(true);
        setErrorMessage(null);

        const countryCode = selectedCityData.country || "XX";
        const countryName = countryMap.current[countryCode] || countryCode;

        api.post('/outlets/import_from_url', {
            url: importUrl,
            city: selectedCityData.name,
            country: countryName,
            lat: parseFloat(selectedCityData.lat),
            lng: parseFloat(selectedCityData.lon),
            instructions: importInstructions || undefined
        })
            .then(res => {
                const data = res.data;
                if (Array.isArray(data)) {
                    // Check if new items were added
                    const prevCount = selectedCityOutlets.length;
                    const newCount = data.length;

                    setSelectedCityOutlets(data);
                    setImportUrl('');
                    setImportInstructions('');
                    setShowAddForm(false);
                    if (data.length > 0 && !discoveredCities.includes(selectedCityData.name)) {
                        setDiscoveredCities(prev => [...prev, selectedCityData.name]);
                    }

                    // Feedback
                    if (newCount > prevCount) {
                        // Logic to highlight? For now just rely on Manual highlight
                    } else {
                        alert('Import completed, but no new unique outlets were added (duplicates skipped).');
                    }
                }
            })
            .catch(err => {
                setErrorMessage(err.response?.data?.detail || err.message);
            })
            .finally(() => setIsImporting(false));
    };

    // Helper to update a single article's date in state
    const updateLocalArticleDate = (targetUrl: string, dateStr: string) => {
        setDigestData((prev: any) => {
            if (!prev) return prev;

            // Calculate cutoff based on active configuration
            const timeframe = selectedTimeframe || "24h";
            const now = new Date();
            const cutoff = new Date();
            if (timeframe === "3days") cutoff.setDate(now.getDate() - 3);
            else if (timeframe === "1week") cutoff.setDate(now.getDate() - 7);
            else if (timeframe === "1month") cutoff.setDate(now.getDate() - 30);
            else cutoff.setDate(now.getDate() - 1); // 24h default

            const newDate = new Date(dateStr);
            const isFresh = newDate >= cutoff;

            const updatedArticles = prev.articles.map((art: any) => {
                if (art.url === targetUrl) {
                    return {
                        ...art,
                        date_str: dateStr,
                        // Update scores based on freshness logic (mimic backend)
                        relevance_score: isFresh ? Math.max(art.relevance_score, 50) : 0,
                        scores: {
                            ...art.scores,
                            date: isFresh ? 30 : 0,
                            is_fresh: isFresh,
                            is_old: !isFresh
                        }
                    };
                }
                return art;
            });
            return { ...prev, articles: updatedArticles };
        });
    };

    const handleDateExtracted = (dateStr: string) => {
        if (debuggerConfig.url) {
            updateLocalArticleDate(debuggerConfig.url, dateStr);
        }
    };

    const handleRulesUpdated = async (domain: string) => {
        if (!digestData?.articles) return;

        // Find all articles from this domain
        const candidates = digestData.articles.filter((art: any) => {
            try {
                return new URL(art.url).hostname.replace('www.', '') === domain;
            } catch { return false; }
        });

        console.log(`Starting batch update for ${candidates.length} sources from ${domain}...`);
        let successCount = 0;
        let failCount = 0;

        // Trigger updates in background
        const batchSize = 3;
        for (let i = 0; i < candidates.length; i += batchSize) {
            const batch = candidates.slice(i, i + batchSize);
            await Promise.all(batch.map(async (art: any) => {
                try {
                    // Use the Test endpoint to apply the newly saved rule
                    const res = await api.post('/scraper/test', { url: art.url });
                    if (res.data.status === 'success' && res.data.extracted_date) {
                        updateLocalArticleDate(art.url, res.data.extracted_date);
                        successCount++;
                    } else {
                        // Success but no date found (or null)
                        failCount++; // Technically not a failure, but no update.
                    }
                } catch (e: any) {
                    // Expected for dead links or blocked sites (400 Bad Request)
                    const msg = e.response?.data?.detail || e.message;
                    console.warn(`Skipping date update for ${art.url}: ${msg}`);
                    failCount++;
                }
            }));
        }
        console.log(`Batch update complete. Updated: ${successCount}, Skipped/Failed: ${failCount}`);
    };

    const handleAssessArticle = async (article: any) => {
        if (!article.url || !article.title) return null;

        try {
            const res = await api.post('/outlets/assess_article', {
                url: article.url,
                title: article.title,
                content: null // force fetch
            });
            const data = res.data;
            return {
                is_politics: data.is_politics,
                confidence: data.confidence,
                reasoning: data.reasoning,
                labels: data.labels
            };
        } catch (err: any) {
            console.error(err);
            alert(`Assessment process failed: ${err.response?.data?.detail || err.message}`);
            return null;
        }
    };


    // ... (Add manual outlet implementation similar to above using api.post)

    const globeComponent = useMemo(() => (
        <Globe
            ref={globeEl}
            globeImageUrl={mapStyle}
            bumpImageUrl="//unpkg.com/three-globe/example/img/earth-topology.png"
            backgroundImageUrl="//unpkg.com/three-globe/example/img/night-sky.png"

            // Polygons (Borders)
            polygonsData={countries.features}
            polygonAltitude={0.005}
            polygonCapColor={() => 'rgba(0, 0, 0, 0)'}
            polygonSideColor={() => 'rgba(0, 0, 0, 0)'}
            polygonStrokeColor={() => mapStyle.includes('day') ? '#000000' : '#888'}
            // @ts-ignore
            polygonStrokeWidth={mapStyle.includes('day') ? 2 : 0.6}
            onPolygonClick={(d: any) => {
                setExpandedCluster(null); // Click background to close cluster
                setSelectedCountry(d);
                if (globeEl.current) {
                    const centroid = turf.centroid(d);
                    const [lng, lat] = centroid.geometry.coordinates;
                    globeEl.current.pointOfView({ lat, lng, altitude: 0.5 }, 1000);
                }
                if (onCountrySelect) onCountrySelect(d.properties.NAME, d.properties.ISO_A2);
            }}

            // Labels (Initials on Marker)
            labelsData={processedData.points}
            labelLat={(d: any) => d.lat}
            labelLng={(d: any) => d.lng}
            labelText={(d: any) => {
                if (!d.name) return '?';
                // Normalize to base char (e.g. Ș -> S) to ensure rendering support
                const char = d.name.charAt(0).normalize("NFD").replace(/[\u0300-\u036f]/g, "");
                return char.toUpperCase();
            }}
            labelLabel={getTooltip} // Show same tooltip when hovering letter
            onLabelClick={handleMapClick} // Allow clicking the letter
            onLabelHover={(d: any) => {
                // Ensure cursor consistency
                document.body.style.cursor = d ? 'pointer' : 'default';
            }}

            // Use default system font for max compatibility
            labelTypeFace={undefined}
            labelFont={undefined}

            // Radius is "radius", Diameter is 2x. Text height needs to fill diameter.
            // User req: "a tid smaller" -> 1.3
            labelSize={(d: any) => d.radius * 1.3}
            labelColor={(d: any) => d.opacity && d.opacity < 1 ? 'transparent' : 'black'}
            labelDotRadius={0} // Hide the auxiliary dot
            labelAltitude={0.0051} // Micro-offset from pointAltitude (0.005) to avoid Z-fighting but minimize gap
            labelResolution={4} // Sharper rendering
            labelIncludeDot={false}

            // Points (Cities & Clusters)
            pointsData={processedData.points}
            pointLat={(d: any) => d.lat}
            pointLng={(d: any) => d.lng}
            pointColor={(d: any) => {
                // Dynamic Highlight Logic:
                // If a city is selected, dim everything else unless it's the selected one or part of the active cluster?
                // The user mentioned "faint colors" issues. 
                // Let's rely on opacity mostly, but ensure Selected City is Bright.

                if (selectedCityData && d.name === selectedCityData.name) {
                    return '#00FFFF'; // Force Cyan for selected
                }

                // Inject opacity if present
                if (d.opacity !== undefined && d.opacity < 1) {
                    // Simple hex to rgba approximation
                    const c = d.color;
                    if (c.startsWith('#')) {
                        const r = parseInt(c.slice(1, 3), 16);
                        const g = parseInt(c.slice(3, 5), 16);
                        const b = parseInt(c.slice(5, 7), 16);
                        return `rgba(${r},${g},${b},${d.opacity})`;
                    }
                    return d.color;
                }
                return d.color;
            }}
            pointRadius={(d: any) => d.radius}
            pointAltitude={0.005} // Raised slightly to fix generic Raycasting/Click interactions
            pointResolution={32}
            onPointHover={(d: any) => {
                // Optimization: Removed setHoverPoint(d) to prevent expensive re-renders
                // This makes the tooltip "snappy"
                document.body.style.cursor = d ? 'pointer' : 'default';
            }}
            onPointClick={(d: any) => {
                if (d) handleMapClick(d);
            }}
            pointLabel={getTooltip}

            // Rings (Visual cues for clusters & Selection Halo)
            ringsData={[
                ...processedData.rings,
                // 1. Temporary Halo for Search Highlight
                ...(highlightedCityId ? (() => {
                    const c = cities.find(x => x.name === highlightedCityId);
                    if (!c) return [];
                    return [{
                        lat: parseFloat(c.lat),
                        lng: parseFloat(c.lng),
                        maxR: 2.5, // Large Halo
                        color: 'rgba(50, 255, 255, 0.8)',
                        propagationSpeed: 5,
                        repeatPeriod: 800
                    }];
                })() : []),
                // 2. Persistent Halo for Selected City
                ...(selectedCityData ? (() => {
                    // Use selectedCityData coordinates directly
                    // Ensure lat/lng are numbers
                    const lat = parseFloat(selectedCityData.lat);
                    const lng = parseFloat(selectedCityData.lng || selectedCityData.lon);
                    if (isNaN(lat) || isNaN(lng)) return [];

                    return [{
                        lat: lat,
                        lng: lng,
                        maxR: 1.5, // Tighter Halo
                        color: 'rgba(50, 200, 255, 0.6)',
                        propagationSpeed: 2,
                        repeatPeriod: 1500
                    }];
                })() : [])
            ]}
            ringLat={(d: any) => d.lat}
            ringLng={(d: any) => d.lng}
            ringMaxRadius={(d: any) => d.maxR}
            ringColor={(d: any) => d.color || 'rgba(255,255,255,0.1)'}
            ringPropagationSpeed={2}
            ringRepeatPeriod={1000}

            // Paths (Spider Legs)
            pathsData={processedData.links}
            // @ts-ignore
            pathPoints={(d: any) => [[d.startLng, d.startLat], [d.endLng, d.endLat]]} // Globe expects [lng, lat]
            pathPointLat={(p: any) => p[1]}
            pathPointLng={(p: any) => p[0]}
            pathColor={(d: any) => d.color}
            pathDashLength={0.1}
            pathDashGap={0.05}
            pathDashAnimateTime={2000}
        />
    ), [mapStyle, countries, processedData, selectedCityData, highlightedCityId, handleMapClick, getTooltip]);

    return (
        <div className="relative w-full h-full bg-slate-950">
            {/* Visual Controls Toggle & Overlay */}
            <div className="absolute bottom-4 left-4 z-20 flex flex-col items-start gap-2">
                {!showControls && (
                    <button
                        onClick={() => setShowControls(true)}
                        className="p-2 rounded-lg border border-slate-700 text-white transition-colors shadow-lg bg-slate-900/80 backdrop-blur hover:bg-slate-800"
                        title="Open Visualization Controls"
                    >
                        <Sliders size={20} />
                    </button>
                )}

                {expandedCluster && (
                    <button
                        onClick={() => setExpandedCluster(null)}
                        className="bg-red-500/80 hover:bg-red-500 text-white p-2 rounded-lg backdrop-blur shadow-lg border border-red-400 animate-in zoom-in duration-200"
                        title="Close Cluster View"
                    >
                        <X size={20} />
                    </button>
                )}

                {showControls && (
                    <div className="bg-slate-900/80 backdrop-blur p-4 rounded-lg border border-slate-700 text-xs text-white space-y-3 w-64 animate-in fade-in slide-in-from-bottom-2 duration-200">
                        <div className="flex items-center justify-between border-b border-slate-700 pb-1 mb-2">
                            <h4 className="font-bold">Viz Controls</h4>
                            <button
                                onClick={() => setShowControls(false)}
                                className="text-slate-400 hover:text-white transition-colors"
                            >
                                <X size={16} />
                            </button>
                        </div>
                        <div>
                            <label className="flex justify-between mb-1">
                                <span>Cluster Radius: {clusterThreshold.toFixed(1)}°</span>
                            </label>
                            <input
                                type="range" min="0.5" max="1.0" step="0.1"
                                value={clusterThreshold}
                                onChange={e => setClusterThreshold(parseFloat(e.target.value))}
                                className="w-full accent-blue-500"
                            />
                        </div>
                        <div>
                            <label className="flex justify-between mb-1">
                                <span>Marker Scale: {markerScale.toFixed(1)}x</span>
                            </label>
                            <input
                                type="range" min="0.4" max="0.8" step="0.05"
                                value={markerScale}
                                onChange={e => setMarkerScale(parseFloat(e.target.value))}
                                className="w-full accent-blue-500"
                            />
                        </div>


                        {/* Map Style Toggle */}
                        <div>
                            <label className="mb-2 block font-semibold text-slate-400">Map Mode</label>
                            <div className="grid grid-cols-2 gap-2">
                                {MAP_STYLES.map(style => (
                                    <button
                                        key={style.name}
                                        onClick={() => setMapStyle(style.url)}
                                        className={`px-2 py-1 rounded text-xs transition-colors border ${mapStyle === style.url
                                            ? 'bg-blue-600 border-blue-500 text-white font-bold'
                                            : 'bg-slate-800 border-slate-600 text-slate-400 hover:bg-slate-700'
                                            }`}
                                    >
                                        {style.name}
                                    </button>
                                ))}
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {/* Debugger Modal */}
            <ScraperDebugger
                isOpen={debuggerConfig.isOpen}
                initialUrl={debuggerConfig.url}
                domain={debuggerConfig.domain}
                onClose={() => setDebuggerConfig(prev => ({ ...prev, isOpen: false }))}
                onDateExtracted={handleDateExtracted}
                onSave={handleRulesUpdated}
            />

            {/* Digest Modal / Full Screen View */}
            {
                digestData && !isGeneratingDigest && (
                    <div className="absolute inset-0 z-50 bg-slate-950/95 backdrop-blur-sm flex items-center justify-center p-8 animate-in fade-in duration-200">
                        <div className="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl w-full max-w-4xl h-[90vh] flex flex-col overflow-hidden">
                            <div className="p-6 border-b border-slate-700 flex justify-between items-center bg-slate-800/50">
                                <div>
                                    <h2 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-indigo-400 bg-clip-text text-transparent">
                                        {selectedCityData?.name} Digest
                                    </h2>
                                    <div className="flex gap-4 mt-2">
                                        <button
                                            onClick={() => setActiveModalTab('articles')}
                                            className={`text-xs font-bold uppercase tracking-wider pb-1 ${activeModalTab === 'articles' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-slate-500'}`}
                                        >
                                            Articles
                                        </button>
                                        <button
                                            onClick={() => setActiveModalTab('digest')}
                                            className={`text-xs font-bold uppercase tracking-wider pb-1 ${activeModalTab === 'digest' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-slate-500'}`}
                                        >
                                            Digest
                                        </button>
                                        <button
                                            onClick={() => setActiveModalTab('analytics')}
                                            className={`text-xs font-bold uppercase tracking-wider pb-1 ${activeModalTab === 'analytics' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-slate-500'}`}
                                        >
                                            Analytics
                                        </button>
                                    </div>
                                </div>


                                <div className="flex gap-2">
                                    <button
                                        onClick={() => setIsTranslateActive(!isTranslateActive)}
                                        className={`flex items-center gap-2 px-3 py-1.5 rounded-full font-bold text-xs transition-colors ${isTranslateActive ? 'bg-amber-500/20 text-amber-400 border border-amber-500/50' : 'bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-white'}`}
                                        title="Toggle Translation"
                                    >
                                        <Languages size={14} />
                                        {isTranslateActive ? 'Original' : 'Translate'}
                                    </button>
                                    <button
                                        onClick={handleDownloadDigest}
                                        className="text-white hover:text-blue-200 bg-slate-700 hover:bg-slate-600 px-4 py-1.5 rounded-full font-bold text-xs transition-colors border border-slate-600"
                                        title="Download as standalone HTML file"
                                    >
                                        📥 Download
                                    </button>
                                    <button
                                        onClick={handleSaveDigest}
                                        disabled={isSaving}
                                        className="text-white hover:text-blue-200 bg-blue-600 hover:bg-blue-500 px-4 py-1.5 rounded-full font-bold text-xs transition-colors disabled:opacity-50"
                                    >
                                        {isSaving ? 'Saving...' : '💾 Save Report'}
                                    </button>
                                    <button
                                        onClick={() => setDigestData(null)}
                                        className="text-slate-400 hover:text-white hover:bg-slate-800 p-2 rounded-full transition-colors"
                                    >
                                        <X size={24} />
                                    </button>
                                </div>
                            </div>

                            <div className="flex-1 overflow-y-auto p-8 custom-scrollbar">
                                {activeModalTab === 'articles' ? (
                                    <div className={`flex flex-col gap-8 ${isTranslateActive ? 'translate-active' : ''}`}>
                                        <div className="border-b border-white/10 pb-4 mb-2">
                                            <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-indigo-400">
                                                News Digest: {selectedCategory}
                                            </h2>
                                            <p className="text-gray-400 text-sm mt-1">
                                                Timeframe: <span className="text-white font-medium">
                                                    {(() => {
                                                        const now = new Date();
                                                        const start = new Date();
                                                        if (selectedTimeframe === '24h') start.setDate(now.getDate() - 1);
                                                        if (selectedTimeframe === '3days') start.setDate(now.getDate() - 3);
                                                        if (selectedTimeframe === '1week') start.setDate(now.getDate() - 7);

                                                        const fmt = (d: Date) => `${d.getDate()}.${d.getMonth() + 1}.${d.getFullYear()}`;
                                                        return `${fmt(start)} - ${fmt(now)}`;
                                                    })()}
                                                </span>
                                            </p>

                                            <div className="mt-4 flex items-center justify-between bg-slate-800/50 p-3 rounded-lg border border-white/5">
                                                <div className="flex items-center gap-4">
                                                    <span className="text-slate-400 text-sm">{selectedArticleUrls.size} articles selected</span>
                                                    <button
                                                        onClick={handleSmartSelect}
                                                        className="px-3 py-1 bg-emerald-900/40 hover:bg-emerald-800/60 text-emerald-400 border border-emerald-800/50 font-bold rounded text-[10px] uppercase tracking-wider transition-all"
                                                        title="Auto-select Fresh & Verified articles"
                                                    >
                                                        Auto-Select
                                                    </button>
                                                </div>
                                                <div className="flex gap-2">
                                                    <button
                                                        onClick={handleGenerateBackendSummary}
                                                        disabled={isSummarizing || selectedArticleUrls.size === 0}
                                                        className="px-4 py-1.5 bg-blue-600 hover:bg-blue-500 text-white font-bold rounded text-xs transition-all disabled:opacity-50 flex items-center gap-2 min-w-[180px] justify-center"
                                                    >
                                                        {isSummarizing ? (
                                                            <>
                                                                <Loader2 className="animate-spin shrink-0" size={14} />
                                                                <span className="truncate max-w-[200px]">{analyzingTickerText || "Analyzing..."}</span>
                                                            </>
                                                        ) : "Summarize Selection"}
                                                    </button>
                                                </div>
                                            </div>
                                        </div>

                                        {/* React Renderer (Virtualization Friendly) */}
                                        <DigestReportRenderer
                                            articles={digestData.articles || []}
                                            category={selectedCategory}
                                            isTranslated={isTranslateActive}
                                            selectedUrls={selectedArticleUrls}
                                            onToggle={handleToggleSelection}
                                            onAssess={handleAssessArticle}
                                            onDebug={handleDebugArticle}
                                        />
                                    </div>
                                ) : activeModalTab === 'digest' ? (
                                    <div className="flex flex-col gap-6">
                                        {digestSummary ? (
                                            <div className="bg-slate-900/50 p-12 rounded-xl border border-white/5 animate-in fade-in duration-500">
                                                <ReactMarkdown
                                                    components={{
                                                        h1: ({ node, ...props }) => <h1 className="text-4xl font-extrabold text-white mb-8 border-b border-white/10 pb-4 mt-8" {...props} />,
                                                        h2: ({ node, ...props }) => <h2 className="text-3xl font-bold text-blue-200 mt-12 mb-6 border-l-4 border-blue-500 pl-4" {...props} />,
                                                        h3: ({ node, ...props }) => <h3 className="text-xl font-bold text-indigo-300 mt-8 mb-3 uppercase tracking-wide" {...props} />,
                                                        p: ({ node, ...props }) => <p className="text-lg text-slate-300 leading-loose mb-6 text-justify" {...props} />,
                                                        ul: ({ node, ...props }) => <ul className="list-disc list-outside ml-6 mb-6 space-y-2 text-slate-300" {...props} />,
                                                        li: ({ node, ...props }) => <li className="pl-2" {...props} />,
                                                        a: ({ node, ...props }) => {
                                                            const href = props.href || '';
                                                            if (href.startsWith('citation:')) {
                                                                const index = parseInt(href.split(':')[1]);
                                                                const article = selectedArticlesList[index - 1]; // 1-based index
                                                                const title = article ? (article.title || 'Source') : `Source ${index}`;
                                                                const url = article ? article.url : '#';

                                                                return (
                                                                    <span className="relative inline-block group mx-1 align-baseline">
                                                                        <a
                                                                            href={url}
                                                                            target="_blank"
                                                                            rel="noopener noreferrer"
                                                                            className="text-emerald-400 font-bold hover:underline cursor-pointer text-sm align-super"
                                                                            onClick={(e) => e.stopPropagation()}
                                                                        >
                                                                            [{index}]
                                                                        </a>
                                                                        {/* Tooltip */}
                                                                        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 p-3 bg-slate-900 border border-slate-700 rounded-lg shadow-xl opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50 text-xs text-left">
                                                                            <div className="font-bold text-white mb-1 line-clamp-2">{title}</div>
                                                                            <div className="text-slate-400 truncate">{new URL(url).hostname}</div>
                                                                        </div>
                                                                    </span>
                                                                );
                                                            }
                                                            return <a className="text-blue-400 hover:text-blue-300 underline underline-offset-4" target="_blank" rel="noopener noreferrer" {...props} />;
                                                        },
                                                        strong: ({ node, ...props }) => <strong className="text-amber-400 font-bold" {...props} />,
                                                        blockquote: ({ node, ...props }) => <blockquote className="border-l-4 border-slate-600 pl-4 italic text-slate-400 my-6 bg-slate-800/30 py-2 rounded-r" {...props} />
                                                    }}
                                                >
                                                    {/* Regex to convert [n] to [[n]](citation:n) */}
                                                    {digestSummary.replace(/\[(\d+)\]/g, '[[/$1/]](citation:$1)').replace(/\[\/(\d+)\/\]/g, '$1')}
                                                </ReactMarkdown>
                                            </div>
                                        ) : (
                                            <div className="text-center py-20 text-slate-600 italic border-2 border-dashed border-slate-800 rounded-xl">
                                                Select articles from the "Articles" tab and click Summarize.
                                            </div>
                                        )}
                                    </div>
                                ) : (
                                    <div className="min-h-full flex flex-col">
                                        <div className="flex justify-between items-center mb-6">
                                            <div className="flex flex-col">
                                                <h3 className="text-xl font-bold text-slate-300 flex items-center gap-2">
                                                    Semantic Cloud
                                                </h3>
                                                <span className="text-slate-500 text-sm font-normal">
                                                    {(() => {
                                                        const now = new Date();
                                                        const start = new Date();
                                                        if (selectedTimeframe === '24h') start.setDate(now.getDate() - 1);
                                                        if (selectedTimeframe === '3days') start.setDate(now.getDate() - 3);
                                                        if (selectedTimeframe === '1week') start.setDate(now.getDate() - 7);

                                                        const fmt = (d: Date) => `${d.getDate()}.${d.getMonth() + 1}`;
                                                        return `${fmt(start)} - ${fmt(now)}`;
                                                    })()}
                                                </span>
                                            </div>

                                            <div className="flex items-center gap-3">
                                                {/* Controls */}
                                                {analyticsKeywords.length > 0 && (
                                                    <>
                                                        <div className="flex bg-slate-800 rounded-lg p-0.5 border border-slate-700">
                                                            <button
                                                                onClick={() => setIsAnalyticsTranslated(!isAnalyticsTranslated)}
                                                                className={`px-3 py-1 text-xs font-bold rounded-md transition-all ${isAnalyticsTranslated ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-white'}`}
                                                                title="Translate to English"
                                                            >
                                                                A/文
                                                            </button>
                                                        </div>

                                                        <div className="flex bg-slate-800 rounded-lg p-0.5 border border-slate-700">
                                                            <button
                                                                onClick={() => setAnalyticsViewMode('cloud')}
                                                                className={`px-3 py-1 text-xs font-bold rounded-md transition-all ${analyticsViewMode === 'cloud' ? 'bg-slate-600 text-white shadow' : 'text-slate-400 hover:text-white'}`}
                                                            >
                                                                ☁️
                                                            </button>
                                                            <button
                                                                onClick={() => setAnalyticsViewMode('columns')}
                                                                className={`px-3 py-1 text-xs font-bold rounded-md transition-all ${analyticsViewMode === 'columns' ? 'bg-slate-600 text-white shadow' : 'text-slate-400 hover:text-white'}`}
                                                            >
                                                                |||
                                                            </button>
                                                        </div>

                                                        <button
                                                            onClick={handleGenerateAnalytics}
                                                            disabled={isAnalyzing}
                                                            className="px-3 py-1 text-xs font-bold text-slate-400 hover:text-white flex items-center gap-1 transition-colors border border-slate-700 rounded-lg bg-slate-800/50"
                                                        >
                                                            <Sparkles className="w-3 h-3" /> Regenerate
                                                        </button>
                                                    </>
                                                )}
                                            </div>
                                        </div>

                                        {isAnalyzing ? (
                                            <div className="flex flex-col items-center justify-center h-64 text-slate-400 animate-pulse">
                                                <Loader2 className="w-8 h-8 mb-3 text-purple-500 animate-spin" />
                                                <p>Extracting Insights...</p>
                                            </div>
                                        ) : analyticsKeywords.length === 0 ? (
                                            <div className="flex flex-col items-center justify-center h-64 border-2 border-dashed border-slate-700/50 rounded-xl bg-slate-900/30">
                                                <p className="text-slate-400 mb-6 max-w-sm text-center">
                                                    Analyze selected articles to detect key entities, sentiment, and patterns.
                                                </p>
                                                <button
                                                    onClick={handleGenerateAnalytics}
                                                    className="bg-purple-600 hover:bg-purple-500 text-white px-6 py-2.5 rounded-full font-bold shadow-lg shadow-purple-900/20 transition-all hover:scale-105 flex items-center gap-2"
                                                >
                                                    <Sparkles className="w-4 h-4" /> Generate Smart Cloud
                                                </button>
                                            </div>
                                        ) : (
                                            analyticsViewMode === 'cloud' ? (
                                                <div className="flex flex-wrap gap-3 content-start justify-center py-4">
                                                    {analyticsKeywords.map((kw: any, i: number) => {
                                                        const displayWord = (isAnalyticsTranslated && kw.translation) ? kw.translation : kw.word;
                                                        // Size logic: 1 to 2.5rem based on score (0-100)
                                                        const scale = 0.8 + ((kw.importance || 50) / 100) * 1.5;

                                                        let bg = "bg-slate-800 border-slate-600 text-slate-300";
                                                        if (kw.sentiment === 'Positive') bg = "bg-green-950/40 border-green-600/50 text-green-300 shadow-[0_0_10px_rgba(34,197,94,0.1)]";
                                                        if (kw.sentiment === 'Negative') bg = "bg-red-950/40 border-red-600/50 text-red-300 shadow-[0_0_10px_rgba(239,68,68,0.1)]";
                                                        if (kw.importance > 85) bg += " ring-1 ring-white/20 font-bold";

                                                        return (
                                                            <div
                                                                key={i}
                                                                className={`relative group cursor-help px-4 py-2 rounded-xl border ${bg} transition-all duration-300 hover:scale-110 hover:shadow-xl hover:z-20`}
                                                                style={{ fontSize: `${Math.max(0.75, scale)}rem` }}
                                                                onMouseEnter={(e) => {
                                                                    if (hoverTimeout.current) clearTimeout(hoverTimeout.current);
                                                                    const rect = e.currentTarget.getBoundingClientRect();
                                                                    const align = rect.left < 200 ? 'left' : (rect.right > window.innerWidth - 300 ? 'right' : 'center');
                                                                    const placement = rect.top < 300 ? 'bottom' : 'top';
                                                                    setActiveTooltip({ word: kw.word, align, placement });
                                                                }}
                                                                onMouseLeave={() => {
                                                                    hoverTimeout.current = setTimeout(() => setActiveTooltip(null), 150);
                                                                }}
                                                            >
                                                                {displayWord}

                                                                {/* Interactive Tooltip */}
                                                                {activeTooltip && activeTooltip.word === kw.word && (
                                                                    <div
                                                                        className={`absolute w-80 bg-slate-900/95 backdrop-blur-md border border-slate-500/50 rounded-xl shadow-2xl p-4 z-[100] text-xs text-left cursor-auto animate-in fade-in zoom-in-95 duration-200 
                                                                            ${activeTooltip.placement === 'bottom' ? 'top-[calc(100%+8px)]' : 'bottom-[calc(100%+8px)]'}
                                                                            ${activeTooltip.align === 'left' ? 'left-0' : activeTooltip.align === 'right' ? 'right-0' : 'left-1/2 -translate-x-1/2'}
                                                                        `}
                                                                        style={{ fontSize: '0.75rem' }}
                                                                        onMouseEnter={() => {
                                                                            if (hoverTimeout.current) clearTimeout(hoverTimeout.current);
                                                                        }}
                                                                        onMouseLeave={() => {
                                                                            hoverTimeout.current = setTimeout(() => setActiveTooltip(null), 150);
                                                                        }}
                                                                    >
                                                                        {/* Invisible Bridge */}
                                                                        <div className={`absolute left-0 w-full h-4 bg-transparent ${activeTooltip.placement === 'bottom' ? '-top-3' : '-bottom-3'}`}></div>

                                                                        <div className="flex justify-between items-start border-b border-slate-700/50 pb-2 mb-3">
                                                                            <div className="font-bold text-white text-base">{displayWord}</div>
                                                                            <div className="flex flex-col items-end gap-1">
                                                                                <span className={`text-[10px] uppercase font-bold px-1.5 py-0.5 rounded ${kw.sentiment === 'Positive' ? 'bg-green-900/50 text-green-400 border border-green-800' : kw.sentiment === 'Negative' ? 'bg-red-900/50 text-red-400 border border-red-800' : 'bg-slate-700/50 text-slate-400 border border-slate-600'}`}>
                                                                                    {kw.sentiment}
                                                                                </span>
                                                                                <span className="text-[10px] text-slate-500 font-mono">Imp: {kw.importance}</span>
                                                                            </div>
                                                                        </div>

                                                                        <div className="space-y-2">
                                                                            <div className="text-slate-400 text-[10px] font-bold uppercase tracking-wider mb-1">
                                                                                Found in {kw.sources?.length || 0} sources
                                                                            </div>
                                                                            <ul className="space-y-2 max-h-40 overflow-y-auto custom-scrollbar pr-1">
                                                                                {kw.sources && kw.sources.length > 0 ? (
                                                                                    kw.sources.map((src: any, idx: number) => (
                                                                                        <li key={idx} className="flex flex-col gap-0.5 bg-slate-800/30 p-2 rounded hover:bg-slate-800/50 transition-colors">
                                                                                            <a href={src.url} target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:text-blue-300 hover:underline font-medium line-clamp-2 leading-tight">
                                                                                                {src.title}
                                                                                            </a>
                                                                                            <span className="text-[10px] text-slate-500">{src.source}</span>
                                                                                        </li>
                                                                                    ))
                                                                                ) : (
                                                                                    <li className="text-slate-600 italic">No direct sources mapped.</li>
                                                                                )}
                                                                            </ul>
                                                                        </div>
                                                                    </div>
                                                                )}
                                                            </div>
                                                        );
                                                    })}
                                                </div>
                                            ) : (
                                                <div className="grid grid-cols-3 gap-0 h-full border border-slate-700 rounded-xl overflow-hidden bg-slate-900/50 min-h-[500px]">
                                                    {['Positive', 'Neutral', 'Negative'].map((sent) => {
                                                        const sentimentKeywords = analyticsKeywords.filter((k: any) => k.sentiment === sent).sort((a: any, b: any) => b.importance - a.importance);
                                                        const headerColor = sent === 'Positive' ? 'text-green-400 border-green-900/30' : sent === 'Negative' ? 'text-red-400 border-red-900/30' : 'text-slate-400 border-slate-700/30';

                                                        return (
                                                            <div key={sent} className="flex flex-col border-r border-slate-700/50 last:border-r-0">
                                                                <div className={`p-4 border-b ${headerColor} bg-slate-900/80 backdrop-blur sticky top-0 z-10 font-bold uppercase tracking-wider text-center text-sm`}>
                                                                    {sent}
                                                                </div>
                                                                <div className="p-4 space-y-3 overflow-y-auto custom-scrollbar flex-1">
                                                                    {sentimentKeywords.map((kw: any, i: number) => (
                                                                        <div key={i} className="bg-slate-800/40 p-3 rounded border border-white/5 hover:bg-slate-800 transition-colors group relative">
                                                                            {/* Tooltip for Column View */}
                                                                            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 p-3 bg-slate-900 border border-slate-700 rounded-lg shadow-xl opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50 text-xs text-left">
                                                                                <div className="font-bold text-white mb-1">Sources:</div>
                                                                                <ul className="space-y-1">
                                                                                    {kw.sources?.slice(0, 3).map((src: any, idx: number) => (
                                                                                        <li key={idx} className="line-clamp-1 text-slate-400">• {src.title || src.url}</li>
                                                                                    ))}
                                                                                </ul>
                                                                            </div>
                                                                            <div className="flex justify-between items-start mb-1">
                                                                                <div className="font-bold text-slate-200">
                                                                                    {isAnalyticsTranslated && kw.translation ? kw.translation : kw.word}
                                                                                </div>
                                                                                <span className="text-xs text-slate-500 font-mono">{kw.importance}</span>
                                                                            </div>
                                                                            {isAnalyticsTranslated && kw.translation && kw.translation !== kw.word && (
                                                                                <div className="text-xs text-slate-500 italic mb-2">{kw.word}</div>
                                                                            )}
                                                                            <div className="flex flex-wrap gap-1 mt-2">
                                                                                {kw.sources?.slice(0, 3).map((src: any, idx: number) => (
                                                                                    <a key={idx} href={src.url} target="_blank" title={src.title} className="w-1.5 h-1.5 rounded-full bg-slate-600 hover:bg-blue-400 transition-colors block"></a>
                                                                                ))}
                                                                            </div>
                                                                        </div>
                                                                    ))}
                                                                </div>
                                                            </div>
                                                        );
                                                    })}
                                                </div>
                                            )
                                        )}
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                )}

            {globeComponent}
            {
                showOutletPanel && (
                    <div className="absolute top-20 left-4 w-96 bg-slate-900/95 backdrop-blur border border-slate-700 rounded-lg shadow-2xl overflow-hidden flex flex-col max-h-[calc(100vh-140px)]">
                        <div className="p-4 border-b border-slate-700 bg-slate-900">
                            {/* Search Bar */}
                            <div className="relative mb-4 max-w-[82%]">
                                <input
                                    type="text"
                                    placeholder="Search City..."
                                    value={searchQuery}
                                    onChange={e => setSearchQuery(e.target.value)}
                                    className="w-full bg-slate-800 border border-slate-600 rounded px-3 py-1.5 text-sm text-white focus:ring-2 focus:ring-blue-500 outline-none"
                                />
                                {searchResults.length > 0 && (
                                    <div className="absolute top-full left-0 right-0 bg-slate-800 border border-slate-600 rounded-b mt-1 max-h-40 overflow-y-auto z-50">
                                        {searchResults.map((city, i) => (
                                            <div
                                                key={i}
                                                onClick={() => handleSearchSelect(city)}
                                                className="px-3 py-2 text-sm text-slate-200 hover:bg-slate-700 cursor-pointer border-b border-slate-700/50 last:border-0"
                                            >
                                                <span className="font-bold">{city.name}</span> <span className="text-slate-500 text-xs">({city.country})</span>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>

                            <div className="flex justify-between items-start mb-3">
                                <div className="flex flex-col gap-1 items-start w-full pr-8">
                                    <h3 className="font-bold text-white text-2xl tracking-tight flex items-baseline gap-3">
                                        {selectedCityName}
                                        {cityInfo && (cityInfo.city_native_name || cityInfo.city_phonetic_name) && (
                                            <span className="text-sm font-normal text-slate-400 font-serif italic">
                                                {cityInfo.city_native_name} {cityInfo.city_phonetic_name && `(${cityInfo.city_phonetic_name})`}
                                            </span>
                                        )}
                                    </h3>
                                    <div className="flex items-center gap-2 flex-wrap">
                                        {cityInfo?.country_flag_url ? (
                                            <img
                                                src={cityInfo.country_flag_url}
                                                alt="Country Flag"
                                                className="h-5 w-auto object-contain rounded-sm shadow-sm"
                                            />
                                        ) : (
                                            cityInfo?.flag_url && (
                                                <img
                                                    src={cityInfo.flag_url}
                                                    alt="Flag"
                                                    className="h-5 w-auto object-contain rounded-sm"
                                                />
                                            )
                                        )}
                                        <span className="text-sm font-bold text-slate-400">
                                            {cityInfo?.country_english || selectedCityData?.country || "Country"}
                                        </span>
                                        {(cityInfo?.country_native || cityInfo?.country_phonetic) && (
                                            <span className="text-xs text-slate-500 font-serif italic border-l border-slate-700 pl-2">
                                                {cityInfo.country_native} {cityInfo.country_phonetic && `(${cityInfo.country_phonetic})`}
                                            </span>
                                        )}
                                    </div>
                                </div>

                                <div className="absolute top-4 right-4 flex gap-2">
                                    <button
                                        onClick={handleRediscoverCity}
                                        title="Rediscover Media Landscape"
                                        className="p-1 hover:bg-slate-800 rounded text-slate-500 hover:text-blue-400 transition-colors"
                                    >
                                        🔄
                                    </button>
                                    <button onClick={() => setShowOutletPanel(false)} className="text-gray-400 hover:text-white">✕</button>
                                </div>
                            </div>
                            {cityInfo && (
                                <div className="text-xs text-slate-400 space-y-2 animate-in slide-in-from-left-2 fade-in duration-300">
                                    <div className="flex flex-wrap gap-2 text-[10px] items-center">
                                        <span className="px-1.5 py-0.5 bg-slate-800 rounded text-slate-300 font-mono">
                                            👥 {cityInfo.population}
                                        </span>
                                        {cityInfo.ruling_party && (
                                            <span className="px-1.5 py-0.5 bg-slate-800 rounded text-slate-300">
                                                🏛 {cityInfo.ruling_party}
                                            </span>
                                        )}
                                    </div>
                                    <p className="text-slate-500 italic leading-relaxed border-l-2 border-slate-700 pl-2">
                                        {cityInfo.description}
                                    </p>
                                </div>
                            )}
                        </div>

                        {/* Controls */}
                        <div className="p-4 space-y-4">
                            <div className="flex gap-2 flex-wrap">
                                {CATEGORIES.map(cat => (
                                    <button
                                        key={cat}
                                        onClick={() => setSelectedCategory(cat)}
                                        className={`px-2 py-1 text-xs rounded border ${selectedCategory === cat ? 'bg-blue-600 border-blue-500 text-white' : 'border-slate-700 text-gray-400'}`}
                                    >
                                        {cat}
                                    </button>
                                ))}
                            </div>

                            {/* Timeframe Selector */}
                            <div className="flex gap-2 bg-slate-900/40 p-1 rounded-lg border border-slate-700/50 my-2">
                                {[
                                    { label: '24h', value: '24h' },
                                    { label: '3 Days', value: '3days' },
                                    { label: '7 Days', value: '1week' }
                                ].map((tf) => (
                                    <button
                                        key={tf.value}
                                        onClick={() => setSelectedTimeframe(tf.value)}
                                        className={`flex-1 py-1 text-[10px] uppercase font-bold rounded text-center transition-all ${selectedTimeframe === tf.value
                                            ? 'bg-blue-600/20 text-blue-400 border border-blue-500/50 shadow-[0_0_10px_rgba(59,130,246,0.2)]'
                                            : 'text-slate-500 hover:text-slate-300 hover:bg-slate-800'
                                            }`}
                                    >
                                        {tf.label}
                                    </button>
                                ))}
                            </div>

                            <div className="flex gap-2 relative w-full">
                                <button
                                    onClick={handleGenerateDigest}
                                    disabled={!isGeneratingDigest && selectedOutletIds.length === 0}
                                    className={`w-full font-medium py-3 rounded-lg shadow-lg transition-all text-base flex justify-center items-center gap-2 h-14 min-w-[300px]
                                        ${isGeneratingDigest
                                            ? 'bg-blue-600 hover:bg-blue-700 text-white shadow-blue-900/40 border border-blue-500/30'
                                            : 'bg-blue-600 hover:bg-blue-700 text-white shadow-blue-900/40 border border-blue-500/30'
                                        }
                                    `}
                                >
                                    {isGeneratingDigest ? (
                                        <>
                                            <div className="flex flex-col items-center">
                                                <span className="flex items-center gap-2">
                                                    <Loader2 className="w-5 h-5 animate-spin" />
                                                    Gathering Articles...
                                                </span>
                                                {progressLog && (
                                                    <div className="mt-1 min-w-[250px] text-center">
                                                        <UIMarquee
                                                            text={progressLog}
                                                            maxLength={40}
                                                            className="text-xs text-blue-200 font-mono opacity-80"
                                                        />
                                                    </div>
                                                )}
                                            </div>
                                        </>
                                    ) : (
                                        <>
                                            <FileText className="w-5 h-5" />
                                            Gather Articles
                                        </>
                                    )}
                                </button>

                                {isGeneratingDigest && (
                                    <button
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            handleStopDigest();
                                        }}
                                        title="Stop Generation"
                                        className="absolute -top-2 -right-2 p-1.5 bg-red-500 hover:bg-red-600 text-white rounded shadow-lg transition-transform hover:scale-110 z-20 border-2 border-slate-900 flex items-center justify-center"
                                    >
                                        <div className="h-3 w-3 bg-white rounded-[1px]" />
                                    </button>
                                )}
                            </div>

                            {/* Sidebar Tabs */}

                            <div className="flex border-b border-slate-700 mt-4">
                                <button
                                    onClick={() => setActiveSideTab('sources')}
                                    className={`flex-1 pb-2 text-sm font-bold transition-colors ${activeSideTab === 'sources' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-slate-500 hover:text-slate-300'}`}
                                >
                                    Sources
                                </button>
                                <button
                                    onClick={() => setActiveSideTab('digests')}
                                    className={`flex-1 pb-2 text-sm font-bold transition-colors ${activeSideTab === 'digests' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-slate-500 hover:text-slate-300'}`}
                                >
                                    Saved Digests
                                </button>
                            </div>
                        </div>

                        {/* Results / List */}
                        <div className="flex-1 overflow-y-auto p-4 text-white text-sm custom-scrollbar">
                            {activeSideTab === 'sources' ? (
                                <>
                                    {isDiscovering && <div className="text-center text-blue-400 mb-4 animate-pulse">Discovering landscape...</div>}

                                    {!digestData && !isDiscovering && (
                                        <div className="space-y-2">
                                            {/* Filters & Header */}
                                            <div className="flex items-center justify-between gap-4 mb-4">
                                                <div className="flex items-center gap-2 text-xs">
                                                    <span className="text-gray-400 font-bold uppercase">Show:</span>
                                                    <label className="flex items-center gap-1 cursor-pointer text-slate-300 hover:text-white">
                                                        <input type="checkbox" defaultChecked className="rounded border-slate-600 bg-slate-800 text-blue-500 focus:ring-0"
                                                            onChange={(e) => {
                                                                // TODO: filtering logic state not yet implemented fully, relying on visual scanning for MVP
                                                            }}
                                                        />
                                                        Local
                                                    </label>
                                                    <label className="flex items-center gap-1 cursor-pointer text-slate-300 hover:text-white">
                                                        <input type="checkbox" defaultChecked className="rounded border-slate-600 bg-slate-800 text-blue-500 focus:ring-0" />
                                                        National
                                                    </label>
                                                </div>
                                            </div>

                                            {selectedCityOutlets.length === 0 && (
                                                <div className="text-center py-4 text-gray-500 text-sm">
                                                    <p>No major outlets automatically found.</p>
                                                    <p>Try the Magic Import below!</p>
                                                </div>
                                            )}

                                            {/* Groups: Manual Top, then AI Sorted by Popularity */}
                                            {(() => {
                                                const manual = selectedCityOutlets.filter(o => o.origin === 'manual');
                                                const auto = selectedCityOutlets.filter(o => o.origin !== 'manual').sort((a, b) => (b.popularity || 0) - (a.popularity || 0));

                                                const renderOutlet = (outlet: any) => (
                                                    <div key={outlet.id} className={`group p-3 rounded transition-all border mb-2 ${selectedOutletIds.includes(outlet.id) ? 'bg-slate-800/80 border-blue-500/30' : 'bg-slate-900/50 border-slate-800 opacity-80 hover:opacity-100'}`}>
                                                        <div className="flex justify-between items-start">
                                                            <div className="flex items-center gap-2">
                                                                <input
                                                                    type="checkbox"
                                                                    checked={selectedOutletIds.includes(outlet.id)}
                                                                    onChange={() => toggleOutletSelection(outlet.id)}
                                                                    className="rounded bg-slate-700 border-slate-600 text-blue-500 focus:ring-0 cursor-pointer"
                                                                />
                                                                <div className="flex flex-col">
                                                                    <div className="flex items-center gap-2">
                                                                        <h4 className={`font-bold transition-colors ${selectedOutletIds.includes(outlet.id) ? 'text-blue-100' : 'text-slate-300'}`}>{outlet.name}</h4>
                                                                        {/* Focus Badge */}
                                                                        {outlet.focus === 'National' && <span className="px-1 py-0.5 rounded bg-indigo-900/50 text-indigo-300 text-[9px] uppercase font-bold border border-indigo-700/50">National</span>}
                                                                        {outlet.focus === 'Local and National' && <span className="px-1 py-0.5 rounded bg-purple-900/50 text-purple-300 text-[9px] uppercase font-bold border border-purple-700/50">Mixed</span>}
                                                                    </div>
                                                                    <div className="flex items-center gap-2 mt-0.5">
                                                                        <span className="text-[10px] uppercase font-mono text-slate-500">{outlet.type || 'Media'}</span>
                                                                        {/* Popularity Stars */}
                                                                        {(outlet.popularity > 0) && (
                                                                            <span className="text-[10px] text-amber-500/80 font-mono" title={`Popularity Score: ${outlet.popularity}/10`}>
                                                                                {"★".repeat(Math.min(outlet.popularity, 5))}
                                                                                <span className='opacity-30'>{"★".repeat(Math.max(0, 5 - outlet.popularity))}</span>
                                                                            </span>
                                                                        )}
                                                                    </div>
                                                                </div>
                                                            </div>
                                                            {outlet.origin === 'manual' && <span className="text-[9px] text-emerald-500 font-bold uppercase tracking-wider bg-emerald-900/20 px-1 rounded">Manually Added</span>}
                                                        </div>
                                                        {outlet.url && selectedOutletIds.includes(outlet.id) && (
                                                            <a
                                                                href={outlet.url}
                                                                target="_blank"
                                                                rel="noopener noreferrer"
                                                                className="mt-2 ml-6 block text-xs text-blue-400 hover:text-blue-300 hover:underline truncate"
                                                            >
                                                                {outlet.url}
                                                            </a>
                                                        )}
                                                    </div>
                                                );

                                                return (
                                                    <>
                                                        {manual.map(renderOutlet)}
                                                        {manual.length > 0 && auto.length > 0 && (
                                                            <div className="my-2 border-t border-slate-800 flex items-center justify-center">
                                                                <span className="bg-slate-900 px-2 text-[10px] text-slate-600 uppercase font-bold -mt-2.5">AI Discovered</span>
                                                            </div>
                                                        )}
                                                        {auto.map(renderOutlet)}
                                                    </>
                                                )
                                            })()}
                                            {editingOutletId !== null && ( // This block was moved outside the renderOutlet function
                                                <div className="flex gap-1 animate-in fade-in zoom-in duration-200 w-full">
                                                    <input
                                                        className="bg-slate-900 border border-blue-500 rounded text-xs px-1 py-0.5 flex-1 text-white outline-none"
                                                        value={editUrl}
                                                        onChange={e => setEditUrl(e.target.value)}
                                                        onKeyDown={e => {
                                                            if (e.key === 'Enter') handleUpdateOutlet(editingOutletId);
                                                            if (e.key === 'Escape') setEditingOutletId(null);
                                                        }}
                                                        autoFocus
                                                    />
                                                    <button onClick={() => handleUpdateOutlet(editingOutletId)} className="text-green-400 hover:text-green-300 px-1">✓</button>
                                                    <button onClick={() => setEditingOutletId(null)} className="text-slate-500 hover:text-slate-300 px-1">✕</button>
                                                </div>
                                            )}

                                        </div >
                                    )}

                                    {
                                        !showAddForm && !digestData && (
                                            <div className="p-4 border-t border-slate-700 mt-4">
                                                <button onClick={() => setShowAddForm(true)} className="w-full py-2 bg-slate-800 text-blue-400 rounded text-sm font-bold border border-slate-600 border-dashed hover:border-blue-500 transition-colors">+ Add Source</button>
                                            </div>
                                        )
                                    }

                                    {
                                        showAddForm && (
                                            <div className="p-4 border-t border-slate-700 space-y-2 mt-4">
                                                <input
                                                    className="w-full bg-slate-800 border-slate-600 rounded p-2 text-white text-xs"
                                                    placeholder="https://example.com"
                                                    value={importUrl}
                                                    onChange={e => setImportUrl(e.target.value)}
                                                />
                                                <div className="flex gap-2">
                                                    <button onClick={handleImportUrl} className="flex-1 bg-purple-600 text-white rounded py-1 text-xs font-bold hover:bg-purple-500">Import</button>
                                                    <button onClick={() => setShowAddForm(false)} className="px-3 bg-slate-700 text-gray-300 rounded py-1 text-xs hover:bg-slate-600">Cancel</button>
                                                </div>
                                            </div>
                                        )
                                    }
                                </>
                            ) : (
                                <div className="space-y-2">
                                    {savedDigests.length === 0 ? (
                                        <div className="text-center text-slate-500 py-8">
                                            No saved digests.<br />
                                            <span className="text-xs">Generate one and click Save!</span>
                                        </div>
                                    ) : (
                                        (savedDigests || []).map((digest: any) => {
                                            const end = new Date(digest.created_at);
                                            const start = new Date(end);

                                            if (digest.timeframe === "24h") start.setDate(end.getDate() - 1);
                                            else if (digest.timeframe === "3days") start.setDate(end.getDate() - 3);
                                            else if (digest.timeframe === "1week") start.setDate(end.getDate() - 7);
                                            else if (digest.timeframe === "1month") start.setDate(end.getDate() - 30);

                                            const f = (d: Date) => d.getDate().toString().padStart(2, '0') + "." + (d.getMonth() + 1).toString().padStart(2, '0');
                                            const y = (d: Date) => d.getFullYear();
                                            const dateRange = `${f(start)} - ${f(end)}.${y(end)}`;

                                            const title = digest.title || "";
                                            const cat = digest.category || "";
                                            const showCategory = cat && !title.toLowerCase().includes(cat.toLowerCase());

                                            return (
                                                <div
                                                    key={digest.id}
                                                    onClick={() => handleLoadDigest(digest)}
                                                    className="bg-slate-800/50 hover:bg-slate-800 border border-slate-700 hover:border-blue-500 p-3 rounded cursor-pointer transition-all group"
                                                >
                                                    <div className="flex justify-between items-start mb-1">
                                                        <h4 className="font-bold text-slate-200 line-clamp-1 group-hover:text-blue-400">{title}</h4>
                                                        <button onClick={(e) => handleDeleteDigest(e, digest.id)} className="text-slate-600 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity">✕</button>
                                                    </div>
                                                    <div className="flex flex-col gap-0.5 text-[10px] text-slate-500 font-medium mt-1">
                                                        <div className="flex items-center gap-2 uppercase font-bold">
                                                            {digest.city && <span className="text-blue-400">{digest.city}</span>}
                                                            {showCategory && <span>• {cat}</span>}
                                                        </div>
                                                        <div className="text-slate-400">
                                                            {dateRange}
                                                        </div>
                                                    </div>
                                                </div>
                                            );
                                        }))}
                                </div>
                            )}
                        </div>
                    </div>
                )
            }


            {
                debugTarget && (
                    <ScraperDebugger
                        isOpen={scraperDebuggerOpen}
                        onClose={() => setScraperDebuggerOpen(false)}
                        initialUrl={debugTarget.url}
                        domain={debugTarget.domain}

                        onSave={handleRuleSaved}
                        onSaving={handleRuleSaving}
                    />
                )
            }
        </div >

    );
}
