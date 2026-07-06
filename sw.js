const CACHE_NAME = 'pose-ai-cache-v1';
const ASSETS = [
  './',
  './index.html',
  './style.css',
  './app.js',
  './manifest.json',
  './icon.png'
];

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      // 一部のリソースが取得できない場合でもエラーで止まらないように個別キャッシュ
      return Promise.allSettled(
        ASSETS.map(asset => 
          cache.add(asset).catch(err => console.warn(`Failed to cache asset: ${asset}`, err))
        )
      );
    })
  );
});

self.addEventListener('fetch', (e) => {
  e.respondWith(
    caches.match(e.request).then((response) => {
      return response || fetch(e.request);
    })
  );
});
