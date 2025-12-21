import time
import psutil
from prometheus_client import start_http_server, Gauge

# Definisi Metriks untuk Prometheus
# Gauge cocok untuk nilai yang bisa naik-turun (kayak CPU/RAM)
CPU_USAGE = Gauge('system_cpu_usage', 'System CPU usage in percent')
RAM_USAGE = Gauge('system_ram_usage', 'System RAM usage in percent')
REQUEST_COUNT = Gauge('http_requests_total', 'Total HTTP requests processed')

def collect_metrics():
    """Fungsi untuk update angka metrics secara realtime"""
    while True:
        # Ambil data CPU & RAM laptop
        cpu = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory().percent
        
        # Set nilai ke metrics Prometheus
        CPU_USAGE.set(cpu)
        RAM_USAGE.set(ram)
        
        # Simulasi request count (nanti bisa diganti real kalau connect ke API)
        # Di sini kita buat statis atau random dulu biar grafik jalan
        REQUEST_COUNT.inc() 
        
        time.sleep(2) # Update setiap 2 detik

if __name__ == '__main__':
    # Jalankan server metrics di port 8000
    print("Prometheus Exporter berjalan di port 8000...")
    start_http_server(8000)
    collect_metrics()