DTC=dtc

SRCDIR=./overlays
OUTDIR=./dtbos

SRC=$(wildcard $(SRCDIR)/*.dts)
TARGET=$(patsubst $(SRCDIR)/%.dts, $(OUTDIR)/%.dtbo, $(SRC))

#Use the C preprocessor to resolve the macros in the overlay file
all: $(TARGET)
	touch $(TARGET)

$(OUTDIR)/%.dtbo: $(SRCDIR)/%.dts
	$(DTC) -@ -I dts -O dtb -o $@ $<
