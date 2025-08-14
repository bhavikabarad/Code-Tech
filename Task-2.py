import csv
from fpdf import FPDF

# Read data from CSV
data = []
with open('petrol.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        data.append(row)

# Extract city-wise average prices
cities = ['Delhi', 'Kolkata', 'Mumbai', 'Chennai']
averages = {}

for city in cities:
    prices = []
    for row in data:
        try:
            prices.append(float(row[city]))
        except:
            continue
    if prices:
        averages[city] = sum(prices) / len(prices)

# Create PDF Report
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)

pdf.cell(200, 10, txt="Petrol Price Report", ln=True, align='C')
pdf.ln(10)

pdf.cell(200, 10, txt="Average Prices by City:", ln=True)
for city, avg in averages.items():
    pdf.cell(200, 10, txt=f"{city}: Rs. {avg:.2f}", ln=True)  # <-- replaced ₹ with Rs.

# Save PDF
pdf.output("petrol_report.pdf")
print("PDF report generated successfully!")
