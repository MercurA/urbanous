'use client';

import { useState, useEffect, useRef, useMemo } from 'react';
import dynamic from 'next/dynamic';
import * as turf from '@turf/turf';
import api from '@/lib/api'; // Use the axios instance with Auth interceptor
import { Sliders, X, Wrench, Search, Map, Calendar, Settings, ChevronDown, ChevronRight, List, Info, Loader2, FileText } from 'lucide-react';
import ScraperDebugger from './ScraperDebugger';
import ReactMarkdown from 'react-markdown';

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

    // Discovery Features
    const [discoveredCities, setDiscoveredCities] = useState<string[]>([]);
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
    const [activeModalTab, setActiveModalTab] = useState<'report' | 'analytics'>('report');

    useEffect(() => {
        if (activeSideTab === 'digests') {
            fetchSavedDigests();
        }
    }, [activeSideTab]);

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
                                    (trigger.previousElementSibling as HTMLElement).style.color = isFresh ? "#4ade80" : "#7f1d1d";
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
                summary_markdown: digestData.digest,
                articles: digestData.articles,
                analysis_source: digestData.analysis_source
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

    const handleGenerateDigest = async () => {
        if (selectedOutletIds.length === 0) {
            setErrorMessage("Please select at least one outlet.");
            return;
        }
        setIsGeneratingDigest(true);
        setErrorMessage(null);
        setDigestData(null);
        setProgressLog('Connecting to stream...');

        try {
            // Retrieve token from storage/api helper if needed
            const token = localStorage.getItem('token');

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
                })
            });

            if (!response.body) throw new Error("No stream body");

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let done = false;
            let buffer = '';

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
                            setProgressLog(`> ${msg.message}`);
                        } else if (msg.type === 'result') {
                            const data = msg.payload;
                            console.log("Received Digest Result:", data); // Debug
                            setDigestData(data);
                            setActiveModalTab('report');
                            setShowOutletPanel(true);

                            if (data.analysis_source) {
                                const hasRateLimit = data.analysis_source.some((k: any) => k.type === "System:RateLimit");
                                if (hasRateLimit) {
                                    alert("⚠️ Rate Limit Warning: Some articles could not be fully analyzed because the AI quota was exceeded. The digest is partial.");
                                }
                            }
                        } else if (msg.type === 'error') {
                            setErrorMessage(msg.message);
                        }
                    } catch (e) {
                        console.warn("Stream parse error for line:", line.substring(0, 50) + "...", e);
                    }
                }
            }

        } catch (err: any) {
            console.error("Digest generation failed", err);
            setErrorMessage(err.message || 'Failed to generate digest');
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
        setActiveModalTab('report');
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
            .then(res => setDiscoveredCities(res.data))
            .catch(err => console.error("Failed to load discovered cities", err));
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
        const newClusters: any[] = [];

        sorted.forEach(city => {
            const lat = parseFloat(city.lat);
            const lng = parseFloat(city.lon);
            const pop = parseInt(city.pop || '0');

            city.radius = getPopScale(city.pop);
            city.isCluster = false;

            const isDiscovered = discoveredCities.includes(city.name);
            const isCapital = pop === countryMaxPop[city.country || 'XX'];

            let color = isDiscovered ? '#34d399' : (isCapital ? '#db2777' : '#64748b');
            city.color = color;
            city.isCapital = isCapital;

            // Find existing cluster
            const existing = newClusters.find(c => {
                if (c.country !== city.country) return false;
                const dist = Math.sqrt(Math.pow(c.lat - lat, 2) + Math.pow(c.lng - lng, 2));
                return dist < clusterThreshold;
            });

            if (existing) {
                existing.subPoints.push(city);
                existing.pop += pop;
                existing.radius = getPopScale(existing.pop) * 1.2;
                existing.isCluster = true;
                if (!existing.isCapital) {
                    existing.color = '#7c3aed';
                }
            } else {
                newClusters.push({
                    ...city,
                    lat: lat,
                    lng: lng,
                    lon: lng,
                    subPoints: [],
                    id: `c-${lat}-${lng}`,
                    pop: pop,
                    isCapital: isCapital
                });
            }
        });
        return newClusters;
    }, [cities, discoveredCities, clusterThreshold, markerScale]);

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

                    items.forEach((item: any, idx: number) => {
                        const isCenter = idx === 0;
                        const r = isCenter ? 0 : spreadFactor * Math.sqrt(idx);
                        const theta = idx * goldenAngle;
                        const exLat = c.lat + (isCenter ? 0 : Math.sin(theta) * r);
                        const exLng = c.lng + (isCenter ? 0 : Math.cos(theta) * r);

                        let itemColor = '#a78bfa';
                        if (discoveredCities.includes(item.name)) itemColor = '#34d399';
                        else if (item.isCapital) itemColor = '#db2777';

                        renderPoints.push({
                            ...item,
                            lat: exLat,
                            lng: exLng,
                            lon: exLng,
                            color: itemColor,
                            radius: getPopScale(item.pop),
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
                    renderRings.push({ lat: c.lat, lng: c.lng, maxR: spreadFactor * Math.sqrt(count) * 1.1, color: 'rgba(255,255,255,0.05)' });
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

    const getTooltip = (d: any) => {
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
    };

    const handleMapClick = (d: any) => {
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
    };

    const handleCityClick = (d: any) => {
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
    };

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

    const handleAssessArticle = (url: string, title: string, btnElement: HTMLElement) => {
        // Optimistic UI
        btnElement.textContent = "⏳ Analyzing...";
        btnElement.style.color = "#fbbf24"; // Amber
        // Disable
        btnElement.style.pointerEvents = "none";

        api.post('/outlets/assess_article', { url, title })
            .then(res => {
                const data = res.data;
                const isPolitics = data.is_politics;
                const color = isPolitics ? '#4ade80' : '#f87171'; // Green : Red
                const icon = isPolitics ? '✅' : '❌';

                // Update Button to Result
                btnElement.innerHTML = `${icon} ${data.confidence}%`;
                btnElement.style.borderColor = color;
                btnElement.style.color = color;

                // Show Verdict Div
                const verdictDiv = btnElement.nextElementSibling as HTMLElement;
                if (verdictDiv) {
                    verdictDiv.style.display = 'block';
                    verdictDiv.style.color = color;
                    verdictDiv.innerHTML = `<b>${isPolitics ? "POLITICS" : "IGNORED"}</b>: ${data.reasoning} <br/> <span style="opacity:0.7; font-size:0.6rem;">[${data.labels.join(', ')}]</span>`;
                }
            })
            .catch(err => {
                btnElement.textContent = "Error";
                btnElement.style.pointerEvents = "auto";
                console.error(err);
            });
    };

    // ... (Add manual outlet implementation similar to above using api.post)

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

            {/* Digest Modal / Full Screen View */}
            {
                digestData && (
                    <div className="absolute inset-0 z-50 bg-slate-950/95 backdrop-blur-sm flex items-center justify-center p-8 animate-in fade-in duration-200">
                        <div className="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl w-full max-w-4xl h-[90vh] flex flex-col overflow-hidden">
                            <div className="p-6 border-b border-slate-700 flex justify-between items-center bg-slate-800/50">
                                <div>
                                    <h2 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-indigo-400 bg-clip-text text-transparent">
                                        {selectedCityData?.name} Digest
                                    </h2>
                                    <div className="flex gap-4 mt-2">
                                        <button
                                            onClick={() => setActiveModalTab('report')}
                                            className={`text-xs font-bold uppercase tracking-wider pb-1 ${activeModalTab === 'report' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-slate-500'}`}
                                        >
                                            Report
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
                                {activeModalTab === 'report' ? (
                                    <div className="flex flex-col gap-8">
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
                                        </div>

                                        {/* HTML Digest Table (Generated by Backend) */}
                                        <div
                                            className="prose prose-invert max-w-none prose-sm prose-headings:text-blue-300 prose-a:text-blue-400 hover:prose-a:text-blue-300 prose-td:align-top"
                                            dangerouslySetInnerHTML={{ __html: digestData.digest }}
                                            onClick={(e) => {
                                                // Event Delegation for Scraper Debugger Triggers injected in HTML
                                                const target = e.target as HTMLElement;

                                                // Scraper Debugger
                                                const trigger = target.closest('.scraper-debug-trigger');
                                                if (trigger) {
                                                    e.preventDefault();
                                                    e.stopPropagation();
                                                    const url = trigger.getAttribute('data-url');
                                                    if (url) handleOpenDebugger(url);
                                                }

                                                // Politics Assessment
                                                const assessTrigger = target.closest('.politics-assess-trigger');
                                                if (assessTrigger) {
                                                    e.preventDefault();
                                                    e.stopPropagation();
                                                    const url = assessTrigger.getAttribute('data-url');
                                                    const title = assessTrigger.getAttribute('data-title');
                                                    if (url && title) {
                                                        const btn = assessTrigger as HTMLElement;
                                                        handleAssessArticle(url, title, btn);
                                                    }
                                                }
                                            }}
                                        />
                                    </div>
                                ) : (
                                    <div className="min-h-full flex flex-col">
                                        <div className="flex justify-between items-center mb-6">
                                            <h3 className="text-xl font-bold text-slate-300">
                                                Semantic Cloud <span className="text-slate-500 text-sm font-normal ml-2">({selectedTimeframe})</span>
                                            </h3>
                                            <div className="flex bg-slate-700/50 rounded-lg p-1 border border-slate-600">
                                                <button
                                                    onClick={() => setAnalyticsMode('source')}
                                                    className={`px-3 py-1 text-xs font-bold rounded-md transition-all ${analyticsMode === 'source' ? 'bg-blue-600 text-white shadow-md' : 'text-slate-400 hover:text-white'}`}
                                                >
                                                    From Sources
                                                </button>
                                                <button
                                                    onClick={() => setAnalyticsMode('digest')}
                                                    className={`px-3 py-1 text-xs font-bold rounded-md transition-all ${analyticsMode === 'digest' ? 'bg-purple-600 text-white shadow-md' : 'text-slate-400 hover:text-white'}`}
                                                >
                                                    From Digest
                                                </button>
                                            </div>
                                        </div>

                                        {(() => {
                                            const activeData = analyticsMode === 'source' ? digestData.analysis_source : digestData.analysis_digest;

                                            if (!activeData || activeData.length === 0) {
                                                return <div className="text-slate-500 italic">No analytics data available for this mode.</div>;
                                            }

                                            return (
                                                <div className="flex flex-wrap gap-2 content-start">
                                                    {activeData.map((kw: any, i: number) => {
                                                        // Size logic: 1 to 2rem
                                                        const scale = 0.8 + ((kw.importance - 1) / 100) * 0.6;

                                                        // Color Logic
                                                        let bg = "bg-slate-700/50 border-slate-600 text-slate-300";
                                                        if (kw.sentiment === 'Positive') bg = "bg-green-900/40 border-green-700/50 text-green-300";
                                                        if (kw.sentiment === 'Negative') bg = "bg-red-900/40 border-red-700/50 text-red-300";
                                                        if (kw.importance > 80) bg = "bg-blue-900/40 border-blue-500/50 text-blue-200 font-bold";

                                                        return (
                                                            <div
                                                                key={i}
                                                                className={`relative group cursor-help px-3 py-1.5 rounded-full border ${bg} transition-all hover:scale-105 hover:shadow-lg hover:z-10`}
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
                                                                {kw.word}

                                                                {/* Interactive Tooltip controlled by State */}
                                                                {activeTooltip?.word === kw.word && activeTooltip && (
                                                                    <div
                                                                        className={`absolute w-72 bg-slate-900/95 backdrop-blur border border-slate-500 rounded-xl shadow-2xl p-3 z-[100] text-xs text-left cursor-auto animate-in fade-in duration-200 
                                                                            ${activeTooltip.placement === 'bottom' ? 'top-[calc(100%+8px)]' : 'bottom-[calc(100%+8px)]'}
                                                                            ${activeTooltip.align === 'left' ? 'left-0' : activeTooltip.align === 'right' ? 'right-0' : 'left-1/2 -translate-x-1/2'}
                                                                        `}
                                                                        onMouseEnter={() => {
                                                                            if (hoverTimeout.current) clearTimeout(hoverTimeout.current);
                                                                        }}
                                                                        onMouseLeave={() => {
                                                                            hoverTimeout.current = setTimeout(() => setActiveTooltip(null), 150);
                                                                        }}
                                                                    >
                                                                        {/* Invisible Bridge */}
                                                                        <div className={`absolute left-0 w-full h-4 bg-transparent ${activeTooltip.placement === 'bottom' ? '-top-3' : '-bottom-3'}`}></div>

                                                                        <div className="flex justify-between items-start border-b border-slate-700 pb-2 mb-2">
                                                                            <div className="font-bold text-white text-sm">{kw.word}</div>
                                                                            <div className="flex flex-col items-end">
                                                                                <span className={`text-[10px] uppercase font-bold px-1.5 rounded ${kw.sentiment === 'Positive' ? 'bg-green-900 text-green-400' : kw.sentiment === 'Negative' ? 'bg-red-900 text-red-400' : 'bg-slate-700 text-slate-400'}`}>
                                                                                    {kw.sentiment}
                                                                                </span>
                                                                                <span className="text-[10px] text-slate-500 mt-1">Score: {kw.importance}</span>
                                                                            </div>
                                                                        </div>

                                                                        <div className="space-y-1">
                                                                            <div className="text-slate-400 font-mono text-[10px]">
                                                                                Found in <span className="text-white font-bold">{kw.source_urls?.length || 0}</span> sources
                                                                            </div>
                                                                            {analyticsMode === 'source' && (
                                                                                <div className="mt-2 pt-2 border-t border-slate-800/50">
                                                                                    <div className="mb-1 text-[10px] text-slate-500 font-bold uppercase">Mentions:</div>
                                                                                    <ul className="space-y-1 max-h-32 overflow-y-auto custom-scrollbar">
                                                                                        {(() => {
                                                                                            const uniqueUrls = Array.from(new Set(kw.source_urls || [])) as string[];
                                                                                            if (uniqueUrls.length === 0) return <li className="text-slate-600 italic">No direct links available.</li>;

                                                                                            return uniqueUrls.map((url: string, idx: number) => {
                                                                                                try {
                                                                                                    const domain = new URL(url).hostname.replace('www.', '');
                                                                                                    return (
                                                                                                        <li key={idx} className="line-clamp-1">
                                                                                                            <a href={url} target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:text-blue-300 hover:underline block truncate" title={url}>
                                                                                                                {domain}
                                                                                                            </a>
                                                                                                        </li>
                                                                                                    )
                                                                                                } catch (e) { return null; }
                                                                                            });
                                                                                        })()}
                                                                                    </ul>
                                                                                </div>
                                                                            )}
                                                                        </div>

                                                                        {/* Arrow */}
                                                                        <div className={`absolute left-1/2 -translate-x-1/2 -ml-1 border-4 border-transparent pointer-events-none 
                                                                            ${activeTooltip.placement === 'bottom'
                                                                                ? 'bottom-full -mb-[1px] border-b-slate-500'
                                                                                : 'top-full -mt-[1px] border-t-slate-500'
                                                                            }
                                                                            ${activeTooltip.align === 'left' ? 'left-6' : activeTooltip.align === 'right' ? 'right-6' : 'left-1/2 -translate-x-1/2'}
                                                                        `}></div>
                                                                    </div>
                                                                )}
                                                            </div>
                                                        )
                                                    })}
                                                </div>
                                            );
                                        })()}
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                )
            }
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
                labelText={(d: any) => d.name ? d.name.charAt(0).toUpperCase() : '?'}
                labelLabel={getTooltip} // Show same tooltip when hovering letter
                onLabelClick={handleMapClick} // Allow clicking the letter
                onLabelHover={(d: any) => {
                    // Ensure cursor consistency
                    document.body.style.cursor = d ? 'pointer' : 'default';
                }}

                // Font: Cinzel (Romanic/Ancient look)
                labelTypeFace={undefined} // Clear default JSON if any (not strictly needed if labelFont is used)
                labelFont="//fonts.gstatic.com/s/cinzel/v11/8vIJ7wvpGWzpsvmICx0.woff"

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

            {
                showOutletPanel && (
                    <div className="absolute top-20 left-4 w-96 bg-slate-900/95 backdrop-blur border border-slate-700 rounded-lg shadow-2xl overflow-hidden flex flex-col max-h-[calc(100vh-140px)]">
                        <div className="p-4 border-b border-slate-700 bg-slate-900">
                            {/* Search Bar */}
                            <div className="relative mb-4">
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

                            <div className="flex gap-2">
                                <button
                                    onClick={handleGenerateDigest}
                                    disabled={isGeneratingDigest || selectedOutletIds.length === 0}
                                    className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 rounded-lg shadow-lg shadow-blue-900/40 transition-all text-base border border-blue-500/30 flex justify-center items-center gap-2 h-14 min-w-[300px]"
                                >
                                    {isGeneratingDigest ? (
                                        <>
                                            <div className="flex flex-col items-center">
                                                <span className="flex items-center gap-2">
                                                    <Loader2 className="w-5 h-5 animate-spin" />
                                                    Generare Digest...
                                                </span>
                                                {progressLog && (
                                                    <span className="text-xs text-blue-200 mt-1 font-mono opacity-80 animate-pulse whitespace-nowrap min-w-[250px] text-center">
                                                        {progressLog.length > 40 ? progressLog.substring(0, 40) + '...' : progressLog}
                                                    </span>
                                                )}
                                            </div>
                                        </>
                                    ) : (
                                        <>
                                            <FileText className="w-5 h-5" />
                                            Generate Digest
                                        </>
                                    )}
                                </button>
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
                                        savedDigests.map((digest: any) => (
                                            <div
                                                key={digest.id}
                                                onClick={() => handleLoadDigest(digest)}
                                                className="bg-slate-800/50 hover:bg-slate-800 border border-slate-700 hover:border-blue-500 p-3 rounded cursor-pointer transition-all group"
                                            >
                                                <div className="flex justify-between items-start mb-1">
                                                    <h4 className="font-bold text-slate-200 line-clamp-1 group-hover:text-blue-400">{digest.title}</h4>
                                                    <button onClick={(e) => handleDeleteDigest(e, digest.id)} className="text-slate-600 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity">✕</button>
                                                </div>
                                                <div className="flex justify-between text-[10px] text-slate-500 uppercase font-bold">
                                                    <span>{digest.category}</span>
                                                    <span>{new Date(digest.created_at).toLocaleDateString()}</span>
                                                </div>
                                            </div>
                                        ))
                                    )}
                                </div>
                            )}
                        </div>
                    </div>
                )}


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
        </div>

    );
}
