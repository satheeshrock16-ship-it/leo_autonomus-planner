import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_BASE || 'http://127.0.0.1:8000';

const API = axios.create({
  baseURL: API_BASE,
  timeout: 10000,
});

export default API;
