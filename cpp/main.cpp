#include <SFML/Graphics.hpp>
#include <cmath>
#define WIDTH 800
#define HEIGHT 600
#define N 10001
#define ARCLENGTH 10000

float *linspace(float start, float end, int num)
{
    float *result = (float *)malloc(num * sizeof(float));

    float step = (end - start) / (num - 1);

    for (int i = 0; i < num; i++)
    {
        result[i] = start + i * step;
    }

    return result;
}

void trapezoid_rule(int num_points, float xmin, float xmax, float y[], float *integral)
{
    float h = (xmax - xmin) / num_points;
    integral[0] = 0;

    for (int i = 1; i < num_points; i++)
    {
        integral[i] = integral[i - 1] + (y[i - 1] + y[i]) * h / 2;
    }

    return;
}

// @brief Compute the curvature
float *curvature(float *x, float *y, int num_points)
{
    float dx[num_points];
    float dy[num_points];
    float ddx[num_points];
    float ddy[num_points];
    float kappa[num_points];

    for (int i = 0; i < num_points - 1; i++)
    {
        dx[i] = x[i + 1] - x[i];
        dy[i] = y[i + 1] - y[i];
    }

    for (int i = 0; i < num_points - 2; i++)
    {
        ddx[i] = dx[i + 1] - dx[i];
        ddy[i] = dy[i + 1] - dy[i];
    }

    for (int i = 0; i < num_points - 2; i++)
    {
        kappa[i] = (dx[i] * ddy[i] - dy[i] * ddx[i]) / pow(dx[i] * dx[i] + dy[i] * dy[i], 3 / 2);
    }

    return kappa;
}

class sfLine : public sf::Drawable
{
public:
    sfLine(const sf::Vector2f &point1, const sf::Vector2f &point2) : color(sf::Color::Blue), thickness(5.f)
    {
        sf::Vector2f direction = point2 - point1;
        sf::Vector2f unitDirection = direction / std::sqrt(direction.x * direction.x + direction.y * direction.y);
        sf::Vector2f unitPerpendicular(-unitDirection.y, unitDirection.x);

        sf::Vector2f offset = (thickness / 2.f) * unitPerpendicular;

        vertices[0].position = point1 + offset;
        vertices[1].position = point2 + offset;
        vertices[2].position = point2 - offset;
        vertices[3].position = point1 - offset;

        for (int i = 0; i < 4; ++i)
            vertices[i].color = color;
    }

    void draw(sf::RenderTarget &target, sf::RenderStates states) const
    {
        target.draw(vertices, 4, sf::Quads);
    }

private:
    sf::Vertex vertices[4];
    float thickness;
    sf::Color color;
};

void draw_points(sf::RenderWindow &window, float x[], float y[], int num_points)
{
    for (int i = 0; i < num_points - 1; i++)
    {
        sf::Vector2f p1 = sf::Vector2f(x[i], y[i]);
        sf::Vector2f p2 = sf::Vector2f(x[i + 1], y[i + 1]);
        sfLine line(p1, p2);
        window.draw(line);
    }
}

/*
@ brief Integrate the curvature along the arc length to get the x and y coordinates

@ param ss: arc length
@ param kappa: curvature
@ param theta0: initial tangent angle
@ param x: x coordinates
@ param y: y coordinates
@ param num_points: number of points
 */
void cesaro(float *ss, float *kappa, float theta0, float *x, float *y, int num_points)
{
    float theta[num_points];

    trapezoid_rule(num_points, 0, ss[num_points - 1], kappa, theta);

    float costheta[num_points];
    float sintheta[num_points];

    for (int i = 0; i < num_points; i++)
    {
        costheta[i] = cos(theta[i]);
        sintheta[i] = sin(theta[i]);
    }

    // Compute x and y coordinates
    trapezoid_rule(num_points, 0, ss[num_points - 1], costheta, x);
    trapezoid_rule(num_points, 0, ss[num_points - 1], sintheta, y);

    return;
}

int main()
{

    float pi = M_PI;

    // Initialize the arc length and curvature
    float *ss = linspace(0, ARCLENGTH, N); // arc length
    float kappa[N];                        // curvature
    float x[N];
    float y[N];
    float theta0 = 0; // initial tangent angle
    float spiral_const = 0.25;

    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "SFML works!");

    // Translate origin to left-center and invert y-axis
    // sf::View view(sf::Vector2f(WIDTH / 2, 0.f), sf::Vector2f(WIDTH, -HEIGHT));

    // Translate origin to center and invert y-axis
    sf::View view(sf::Vector2f(0.f, 0.f), sf::Vector2f(WIDTH, -HEIGHT));

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            switch (event.type)
            {
            // window closed
            case sf::Event::Closed:
                window.close();
                break;

            // key pressed
            case sf::Event::KeyPressed:
                if (event.key.code == sf::Keyboard::Left)
                    spiral_const -= 0.01;
                if (event.key.code == sf::Keyboard::Right)
                    spiral_const += 0.01;
                if (event.key.code == sf::Keyboard::Escape)
                    window.close();
                break;

            // we don't process other types of events
            default:
                break;
            }
        }

        window.clear();
        window.setView(view);
        for (int i = 0; i < N; i++)
        {
            if (ss[i] == 0)
            {
                kappa[i] = 0;
                continue;
            }
            kappa[i] = spiral_const / sqrt(ss[i]);
        }

        // Compute x and y coordinates

        cesaro(ss, kappa, theta0, x, y, N);
        draw_points(window, x, y, N);
        window.display();
    }

    free(ss);

    return 0;
}
