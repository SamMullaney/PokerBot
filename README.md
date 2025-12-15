# Poker Vision and Decision-Making Bot

## Project Overview

This project implements an end-to-end poker bot that combines computer vision, optical character recognition, probabilistic modeling, and game-theoretic decision-making. The system is designed to observe an online poker table through screenshots, extract the relevant game state, and make mathematically grounded betting decisions. The project focuses on Texas Hold'em and is structured to be modular, extensible, and suitable for simulation-based evaluation.

The core components of the project include card recognition, pot and bet size detection, state vector construction, poker math utilities, and a hero versus villain simulation framework.

---

## Computer Vision: Card Identification

The first major component of the project is a card recognition system.

A YOLOv8 image classification model was trained to recognize all 52 playing cards. Each card is treated as a separate class. Training data consisted of cropped images of individual cards, organized into train and validation directories following YOLO classification conventions.

Key details:

* Model: YOLOv8 classification (yolov8m-cls)
* Input size: 320x320
* Output: Card label and confidence score
* Accuracy: Approximately 95 to 99 percent top-1 accuracy after optimization

This model is used to classify cropped regions of a table screenshot where cards are known to appear.

---

## Fixed Table Geometry and Card Localization

Because the poker client uses a consistent layout, the locations of all cards on the table are fixed in pixel space.

Manually measured bounding boxes are used for:

* Five community card positions
* Two player hand card positions

For each screenshot:

1. The image is cropped using predefined coordinates.
2. Each crop is passed into the trained card classifier.
3. Low-confidence predictions are filtered and treated as NO_CARD.

This allows the system to correctly handle all stages of the hand including preflop, flop, turn, and river.

---

## State Vector Construction

The vision output is converted into fixed-length numeric state vectors suitable for mathematical modeling and simulation.

* Cards are encoded as integers from 0 to 51.
* Missing cards are encoded as -1.

State representation:

* Player hand vector of length 2
* Community card vector of length 5

Example:

* Player hand: [42, 14]
* Community cards: [28, 31, 20, -1, -1]

This representation is used throughout the simulation and decision logic.

---

## OCR: Pot Size and Bet Size Detection

In addition to cards, the bot must detect numerical values from the table.

An OCR pipeline using Tesseract was implemented to read:

* Current pot size
* Current bet size

Key techniques:

* Precise cropping using fixed screen coordinates
* Image upscaling for improved digit clarity
* Grayscale conversion and adaptive thresholding
* Multiple preprocessing variants per crop
* Strict digit and decimal filtering
* Heuristics to reject implausible OCR outputs

The system correctly handles cases where no number is present, such as preflop or when no bet has been made.

---

## Poker Math and GTO Utilities

A dedicated math module provides the foundation for decision-making. This includes:

* Pot odds
* Minimum Defense Frequency
* Expected value calculations for calls and bets
* Bluff to value ratios by street
* Street-aware bluff frequency
* Draw equity estimation using the Rule of 2 and 4
* Realized equity heuristics based on position and hand class
* Randomized action selection consistent with mixed strategies

These utilities are designed to be reusable and independent of the vision system.

---

## Hero and Villain Strategy Models

Two strategy agents are implemented.

### Villain Model

The villain uses a simplified but realistic poker strategy:

* Bets more frequently with strong hands
* Folds weak hands at reasonable frequencies
* Checks marginal hands
* Occasionally bluffs at controlled rates

The villain is intentionally not optimal and does not adapt. This provides a stable baseline opponent for evaluation.

### Hero Model

The hero implements a more advanced, GTO-inspired strategy:

* Uses equity thresholds for betting, calling, and folding
* Applies street-dependent bluff to value ratios
* Randomizes actions based on mathematically justified frequencies
* Balances aggression and defense using pot odds and MDF
* Bets proactively rather than relying on passive play

The hero logic is modular and designed for continued refinement.

---

## Monte Carlo Simulation Framework

A Monte Carlo simulation engine evaluates the performance of the hero strategy against the villain.

Simulation details:

* Random Texas Hold'em hands
* Full five-card boards
* Pot tracking and betting flow
* Equity calculation using a poker hand evaluator
* Action frequency tracking
* Cumulative profit tracking over thousands of hands

Results are visualized using matplotlib, including:

* Profit over time
* Hero action frequencies
* Villain action frequencies

After refining the decision logic and betting flow, the hero achieves consistently positive expected value against the villain.

---

## Current Capabilities

* Card recognition from screenshots
* Pot and bet size detection via OCR
* Robust handling of missing information
* Conversion of visual input into structured game state
* GTO-inspired betting logic
* Monte Carlo evaluation of strategy performance

---

## Future Extensions

Potential future work includes:

* Multi-street betting sequences
* Opponent profiling and exploitative adjustments
* Stack size detection and all-in logic
* Multi-player simulations
* Reinforcement learning for strategy optimization
* Real-time decision loop integration

---

## Disclaimer

This project is for educational and research purposes only. It is not intended for real-money gambling or use in violation of any platform's terms of service.
