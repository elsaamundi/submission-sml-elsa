import time
import psutil
import random
from prometheus_client import start_http_server, Gauge, Counter

# --- METRIK 1: CPU Usage (Gauge) ---
cpu_usage = Gauge('system_cpu_usage', 'Penggunaan CPU dalam persen')

# --- METRIK 2: RAM Usage (Gauge) ---
ram_usage = Gauge('system_ram_usage', 'Penggunaan RAM dalam persen')

# --- METRIK 3: Request Count (Counter) ---
# Ini buat menuhin syarat "3 Metrik". Kita anggap ini jumlah request ke model.
request_count = Counter('model_request_count', 'Total Request yang masuk ke model')

def collect_metrics():
    # 1. Ambil data CPU Real
    cpu = psutil.cpu_percent(interval=1)
    cpu_usage.set(cpu)
    
    # 2. Ambil data RAM Real
    ram = psutil.virtual_memory().percent
    ram_usage.set(ram)
    
    # 3. Simulasi Request (Biar angkanya nambah terus)
    # Anggaplah ada request masuk secara acak
    if random.random() > 0.5: 
        request_count.inc()
        
    print(f"Update Data -> CPU: {cpu}%, RAM: {ram}%")

if __name__ == '__main__':
    # Jalanin server di port 8000
    start_http_server(8000)
    print("Prometheus Exporter berjalan di port 8000...")
    
    while True:
        collect_metrics()
        # Update setiap 1 detik
        time.sleep(1)