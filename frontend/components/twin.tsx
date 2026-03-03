'use client';

import { useState, useRef, useEffect } from 'react';
import { Send, Bot, User } from 'lucide-react';

interface Message {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    timestamp: Date;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';
const TOKEN_KEY = 'dt_access_token';
const SESSION_KEY = 'dt_session_id';
const USER_PROFILE_KEY = 'dt_user_profile';

interface AuthResponse {
    access_token: string;
    token_type: string;
    user_id: string;
    first_name: string;
    last_name: string;
    email?: string;
    phone?: string;
}

interface RegisterResponse {
    message: string;
    user_id: string;
    email: string;
}

interface UserProfile {
    user_id: string;
    first_name: string;
    last_name: string;
    email?: string;
    phone?: string;
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null;
}

function isRegisterResponse(data: unknown): data is RegisterResponse {
    if (!isRecord(data)) return false;
    return (
        typeof data.message === 'string' &&
        typeof data.user_id === 'string' &&
        typeof data.email === 'string'
    );
}

function isAuthResponse(data: unknown): data is AuthResponse {
    if (!isRecord(data)) return false;
    return (
        typeof data.access_token === 'string' &&
        typeof data.token_type === 'string' &&
        typeof data.user_id === 'string' &&
        typeof data.first_name === 'string' &&
        typeof data.last_name === 'string'
    );
}

export default function Twin() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [sessionId, setSessionId] = useState<string>('');
    const [token, setToken] = useState<string>('');
    const [profile, setProfile] = useState<UserProfile | null>(null);
    const [authMode, setAuthMode] = useState<'login' | 'register'>('login');
    const [firstName, setFirstName] = useState('');
    const [lastName, setLastName] = useState('');
    const [email, setEmail] = useState('');
    const [phone, setPhone] = useState('');
    const [password, setPassword] = useState('');
    const [authError, setAuthError] = useState('');
    const [isAuthLoading, setIsAuthLoading] = useState(false);
    const [authInfo, setAuthInfo] = useState('');
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    useEffect(() => {
        const storedToken = localStorage.getItem(TOKEN_KEY) || '';
        const storedSession = localStorage.getItem(SESSION_KEY) || '';
        const storedProfile = localStorage.getItem(USER_PROFILE_KEY);
        setToken(storedToken);
        setSessionId(storedSession);
        if (storedProfile) {
            setProfile(JSON.parse(storedProfile));
        }
    }, []);

    const handleAuthSuccess = (data: AuthResponse) => {
        setToken(data.access_token);
        localStorage.setItem(TOKEN_KEY, data.access_token);
        const nextProfile: UserProfile = {
            user_id: data.user_id,
            first_name: data.first_name,
            last_name: data.last_name,
            email: data.email,
            phone: data.phone,
        };
        setProfile(nextProfile);
        localStorage.setItem(USER_PROFILE_KEY, JSON.stringify(nextProfile));
        setAuthError('');
        setAuthInfo('');
    };

    const submitAuth = async () => {
        setIsAuthLoading(true);
        setAuthError('');
        setAuthInfo('');
        if (authMode === 'register' && !email.trim()) {
            setAuthError('Email is required');
            setIsAuthLoading(false);
            return;
        }
        try {
            const endpoint = authMode === 'register' ? '/auth/register' : '/auth/login';
            const payload = authMode === 'register'
                ? {
                    first_name: firstName,
                    last_name: lastName,
                    email: email.trim(),
                    phone: phone || undefined,
                    password,
                }
                : {
                    email: email.trim(),
                    password,
                };

            const response = await fetch(`${API_BASE_URL}${endpoint}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            const raw = await response.text();
            let data: unknown = {};
            if (raw) {
                try {
                    data = JSON.parse(raw);
                } catch {
                    data = { detail: raw };
                }
            }
            if (!response.ok) {
                const detail = isRecord(data) && typeof data.detail === 'string'
                    ? data.detail
                    : 'Authentication failed';
                throw new Error(detail);
            }
            if (authMode === 'register') {
                if (!isRegisterResponse(data)) {
                    throw new Error('Unexpected register response from server');
                }
                const registerData = data;
                setAuthInfo(registerData.message || 'Verification email sent.');
                setAuthMode('login');
                setPassword('');
            } else {
                if (!isAuthResponse(data)) {
                    throw new Error('Unexpected login response from server');
                }
                handleAuthSuccess(data);
            }
        } catch (error) {
            const message = error instanceof Error ? error.message : 'Authentication failed';
            setAuthError(message);
        } finally {
            setIsAuthLoading(false);
        }
    };

    const logout = () => {
        setToken('');
        setProfile(null);
        setSessionId('');
        setMessages([]);
        localStorage.removeItem(TOKEN_KEY);
        localStorage.removeItem(SESSION_KEY);
        localStorage.removeItem(USER_PROFILE_KEY);
    };

    const sendMessage = async () => {
        if (!input.trim() || isLoading || !token) return;

        const userMessage: Message = {
            id: Date.now().toString(),
            role: 'user',
            content: input,
            timestamp: new Date(),
        };

        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);

        try {
            const response = await fetch(`${API_BASE_URL}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    Authorization: `Bearer ${token}`,
                },
                body: JSON.stringify({
                    message: input,
                    session_id: sessionId || undefined,
                }),
            });

            if (!response.ok) throw new Error('Failed to send message');

            const data = await response.json();

            if (!sessionId) {
                setSessionId(data.session_id);
                localStorage.setItem(SESSION_KEY, data.session_id);
            }

            const assistantMessage: Message = {
                id: (Date.now() + 1).toString(),
                role: 'assistant',
                content: data.response,
                timestamp: new Date(),
            };

            setMessages(prev => [...prev, assistantMessage]);
        } catch (error) {
            console.error('Error:', error);
            // Add error message
            const errorMessage: Message = {
                id: (Date.now() + 1).toString(),
                role: 'assistant',
                content: 'Sorry, I encountered an error. Please try again.',
                timestamp: new Date(),
            };
            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    return (
        <div className="flex flex-col h-full bg-gray-50 rounded-lg shadow-lg">
            {/* Header */}
            <div className="bg-gradient-to-r from-slate-700 to-slate-800 text-white p-4 rounded-t-lg">
                <div className="flex items-center justify-between gap-3">
                    <h2 className="text-xl font-semibold flex items-center gap-2">
                    <Bot className="w-6 h-6" />
                    AI Digital Twin
                    </h2>
                    {profile && (
                        <button
                            onClick={logout}
                            className="text-xs bg-slate-600 hover:bg-slate-500 px-3 py-1 rounded"
                        >
                            Logout
                        </button>
                    )}
                </div>
                <p className="text-sm text-slate-300 mt-1">
                    {profile ? `Welcome, ${profile.first_name}` : 'Sign in to start chatting'}
                </p>
            </div>

            {!token && (
                <div className="p-4 border-b border-gray-200 bg-white">
                    <div className="flex gap-2 mb-3">
                        <button
                            onClick={() => setAuthMode('login')}
                            className={`px-3 py-1 rounded text-sm ${authMode === 'login' ? 'bg-slate-700 text-white' : 'bg-gray-100 text-gray-700'}`}
                        >
                            Login
                        </button>
                        <button
                            onClick={() => setAuthMode('register')}
                            className={`px-3 py-1 rounded text-sm ${authMode === 'register' ? 'bg-slate-700 text-white' : 'bg-gray-100 text-gray-700'}`}
                        >
                            Register
                        </button>
                    </div>

                    {authMode === 'register' && (
                        <div className="grid grid-cols-2 gap-2 mb-2">
                            <input
                                value={firstName}
                                onChange={(e) => setFirstName(e.target.value)}
                                placeholder="First name"
                                className="px-3 py-2 border rounded"
                            />
                            <input
                                value={lastName}
                                onChange={(e) => setLastName(e.target.value)}
                                placeholder="Last name"
                                className="px-3 py-2 border rounded"
                            />
                        </div>
                    )}

                    <div className="grid grid-cols-2 gap-2 mb-2">
                        <input
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            placeholder={authMode === 'register' ? 'Email (required)' : 'Email'}
                            className="px-3 py-2 border rounded"
                        />
                        <input
                            value={phone}
                            onChange={(e) => setPhone(e.target.value)}
                            placeholder="Phone (optional)"
                            className="px-3 py-2 border rounded"
                            disabled={authMode === 'login'}
                        />
                    </div>
                    <input
                        type="password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        placeholder="Password"
                        className="w-full px-3 py-2 border rounded mb-2"
                    />
                    {authError && <p className="text-red-600 text-sm mb-2">{authError}</p>}
                    {authInfo && <p className="text-green-700 text-sm mb-2">{authInfo}</p>}
                    <button
                        onClick={submitAuth}
                        disabled={isAuthLoading}
                        className="w-full px-3 py-2 bg-slate-700 text-white rounded disabled:opacity-50"
                    >
                        {isAuthLoading ? 'Please wait...' : authMode === 'register' ? 'Create account' : 'Login'}
                    </button>
                </div>
            )}

            {/* Messages */}
            <div className={`flex-1 overflow-y-auto p-4 space-y-4 ${!token ? 'opacity-50 pointer-events-none' : ''}`}>
                {messages.length === 0 && (
                    <div className="text-center text-gray-500 mt-8">
                        <Bot className="w-12 h-12 mx-auto mb-3 text-gray-400" />
                        <p>Hello! I&apos;m your Digital Twin.</p>
                        <p className="text-sm mt-2">Ask me anything about AI deployment!</p>
                    </div>
                )}

                {messages.map((message) => (
                    <div
                        key={message.id}
                        className={`flex gap-3 ${
                            message.role === 'user' ? 'justify-end' : 'justify-start'
                        }`}
                    >
                        {message.role === 'assistant' && (
                            <div className="flex-shrink-0">
                                <div className="w-8 h-8 bg-slate-700 rounded-full flex items-center justify-center">
                                    <Bot className="w-5 h-5 text-white" />
                                </div>
                            </div>
                        )}

                        <div
                            className={`max-w-[70%] rounded-lg p-3 ${
                                message.role === 'user'
                                    ? 'bg-slate-700 text-white'
                                    : 'bg-white border border-gray-200 text-gray-800'
                            }`}
                        >
                            <p className="whitespace-pre-wrap">{message.content}</p>
                            <p
                                className={`text-xs mt-1 ${
                                    message.role === 'user' ? 'text-slate-300' : 'text-gray-500'
                                }`}
                            >
                                {message.timestamp.toLocaleTimeString()}
                            </p>
                        </div>

                        {message.role === 'user' && (
                            <div className="flex-shrink-0">
                                <div className="w-8 h-8 bg-gray-600 rounded-full flex items-center justify-center">
                                    <User className="w-5 h-5 text-white" />
                                </div>
                            </div>
                        )}
                    </div>
                ))}

                {isLoading && (
                    <div className="flex gap-3 justify-start">
                        <div className="flex-shrink-0">
                            <div className="w-8 h-8 bg-slate-700 rounded-full flex items-center justify-center">
                                <Bot className="w-5 h-5 text-white" />
                            </div>
                        </div>
                        <div className="bg-white border border-gray-200 rounded-lg p-3">
                            <div className="flex space-x-2">
                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-100" />
                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-200" />
                            </div>
                        </div>
                    </div>
                )}

                <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="border-t border-gray-200 p-4 bg-white rounded-b-lg">
                <div className="flex gap-2">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={handleKeyPress}
                        placeholder="Type your message..."
                        className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-slate-600 focus:border-transparent text-gray-800"
                        disabled={isLoading}
                    />
                    <button
                        onClick={sendMessage}
                        disabled={!input.trim() || isLoading || !token}
                        className="px-4 py-2 bg-slate-700 text-white rounded-lg hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-slate-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                    >
                        <Send className="w-5 h-5" />
                    </button>
                </div>
            </div>
        </div>
    );
}