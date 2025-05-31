# Rmbd Dev Kit

This directory contains tools for doing development on rmbd itself. It is for use by the developers of rmbd, and is not for use by users of it, and does not ship with the project distribution.

The whole thing about this directory is that it doesn't care about name collisions in its class and function names; it assumes that rmbd is the only thing you're working on in your session. It names functions whatever it wants, and doesn't put them inside packages. (This is intentional, to make it convenient for rmbd developers to call them without having to type out the package name all the time.) So it is _not_ suitable for redistribution to users.


## Naming Rule 

Rule-Based Word Shortening Algorithm
1 Remove all vowels, except the first one if needed for readability. 
2 Keep the first and last consonants intact.
3 Reduce double consonants into a single letter (e.g., Process ? Prcs). 
4 Maintain phonetic recognition so the shortened word remains understandable.
5 Keep critical letters that define meaning (e.g., Calibration ? Clbrtn).
SrvMtr ? Servo Motor

StprMtr ? Stepper Motor
. Actuators
BrshDcMtr ? Brushed DC Motor

BlDcMtr ? Brushless DC Motor

StprMtr ? Stepper Motor

SrvMtr ? Servo Motor

LnActr ? Linear Actuator

HydrActr ? Hydraulic Actuator

PnmActr ? Pneumatic Actuator

2. Transmission Systems
PlntryGbx ? Planetary Gearbox

SprGbx ? Spur Gearbox

WrmGbx ? Worm Gearbox

TmngBlt ? Timing Belt System

FlxCplng ? Flexible Coupling

RgCplng ? Rigid Coupling

MagCplng ? Magnetic Coupling

BlBrng ? Ball Bearing

RlBrng ? Roller Bearing

3. Sensors & Feedback Devices
Encdr ? Encoder

Gyro ? Gyroscope

Accelr ? Accelerometer

LdrScnr ? LiDAR Scanner

TchSnsr ? Tactile Sensor

FrcTrqSnsr ? Force/Torque Sensor

IRRngSnsr ? Infrared Range Sensor

UltrSnsr ? Ultrasonic Sensor

TempSnsr ? Temperature Sensor

OptcFlw ? Optical Flow Sensor

4. Power & Control Components
Rlly ? Relay

TglSw ? Toggle Switch

PushBtnSw ? Push Button Switch

LmtSw ? Limit Switch

DcDcCnvrtr ? DC-DC Converter

AcDcCnvrtr ? AC-DC Converter

Invrtr ? Inverter

PwrSpl ? Power Supply

5. Thermal & Protection Components
Fus ? Fuse

CirBrkr ? Circuit Breaker

HtSnk ? Heat Sink

ClngFn ? Cooling Fan

ThrmSnsr ? Thermal Sensor

This should help categorize your electromechanical components efficiently within your Actuator Management Unit structure. If you need refinements or additional abbreviations, let me know�I'm happy to fine-tune it! ?


1. Battery & Power Sources
LiIonBtr ? Lithium-Ion Battery

NiMHBtr ? Nickel-Metal Hydride Battery

PbAcBtr ? Lead-Acid Battery

SuprCpc ? Supercapacitor (Quick energy bursts)

FuelCll ? Fuel Cell (Hydrogen-powered)

SolrPnl ? Solar Panel (Alternative power)

7. Hydraulic Power Sources
HydrPump ? Hydraulic Pump (Generates fluid pressure)

PwrUnit ? Hydraulic Power Unit (HPU)

HydrAcc ? Hydraulic Accumulator (Stores pressurized fluid)

PrsrRgltr ? Pressure Regulator (Maintains stability)

HydrRes ? Hydraulic Reservoir (Fluid storage)
PnmActr ? Pneumatic Actuator

LnActr ? Linear Actuator

HydrCylndr ? Hydraulic Cylinder

PlntryGbx ? Planetary Gearbox

TmngBlt ? Timing Belt System

DcDcCnvrtr ? DC-DC Converter

Invrtr ? Inverter

Hydr ? Hydraulic Valve
Vlv
PrsrRgltr ? Pressure Regulator