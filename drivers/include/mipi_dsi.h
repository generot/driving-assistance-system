#ifndef __MIPI_DSI__
#define __MIPI_DSI__

//Required macros for MIPI DSI Display DTO.
//RasPI OS Linux Kernel v6.1.0

#define MIPI_DSI_MODE_VIDEO		0
#define MIPI_DSI_MODE_VIDEO_BURST	1
/* Video pulse mode.
 * Not set denotes sync event mode. (DSI spec V1.1 8.11.2)
 */
#define MIPI_DSI_MODE_VIDEO_SYNC_PULSE	2
/* Enable auto vertical count mode */
#define MIPI_DSI_MODE_VIDEO_AUTO_VERT	8
/* Enable hsync-end packets in vsync-pulse and v-porch area */
#define MIPI_DSI_MODE_VIDEO_HSE		16
/* Transmit NULL packets or LP mode during hfront-porch area.
 * Not set denotes sending a blanking packet instead. (DSI spec V1.1 8.11.1)
 */
#define MIPI_DSI_MODE_VIDEO_NO_HFP	32
/* Transmit NULL packets or LP mode during hback-porch area.
 * Not set denotes sending a blanking packet instead. (DSI spec V1.1 8.11.1)
 */
#define MIPI_DSI_MODE_VIDEO_NO_HBP	64
/* Transmit NULL packets or LP mode during hsync-active area.
 * Not set denotes sending a blanking packet instead. (DSI spec V1.1 8.11.1)
 */
#define MIPI_DSI_MODE_VIDEO_NO_HSA	128
/* Flush display FIFO on vsync pulse */
#define MIPI_DSI_MODE_VSYNC_FLUSH	256
/* Disable EoT packets in HS mode. (DSI spec V1.1 8.1)  */
#define MIPI_DSI_MODE_NO_EOT_PACKET	512
/* Device supports non-continuous clock behavior (DSI spec V1.1 5.6.1) */
#define MIPI_DSI_CLOCK_NON_CONTINUOUS	1024
/* Transmit data in low power */
#define MIPI_DSI_MODE_LPM		2048
/* transmit data ending at the same time for all lanes within one hsync */
#define MIPI_DSI_HS_PKT_END_ALIGNED	4096

#define MIPI_DSI_FMT_RGB888 		0

#endif
